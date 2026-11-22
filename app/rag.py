"""
rag.py: answer FAQ questions from OUR OWN documents (RAG = Retrieval-Augmented Generation).

How it works, in 4 steps:
    1. CHUNK:    split each FAQ file in docs/faq/ into small pieces (one "## question" section each).
    2. INDEX:    turn every chunk into a vector and store the vectors in a FAISS index.
    3. RETRIEVE: turn the customer's question into a vector and find the closest chunks.
    4. GENERATE: give ONLY those chunks to the LLM and ask it to answer from them.
                 LLM chain (see llm.py): Groq first, then local Ollama if Groq fails.
                 If no LLM can answer, return the best chunk's text directly ("extractive").

Definitions:
    - RAG: instead of trusting what the LLM "remembers", we first LOOK UP the facts in our
      documents and give them to the LLM. Everyday example: an open-book exam.
    - Chunking: cutting documents into small pieces, so we can retrieve just the relevant
      paragraph instead of a whole document (shorter prompt, more focused answer).
    - FAISS: a library from Meta for fast vector search ("which stored vectors are closest
      to this one?"). With ~40 chunks a plain loop would also be fine; FAISS is what you use
      when you have millions of vectors, and the code stays the same.
    - Grounding: the answer must come from the retrieved text, not from the LLM's memory.
    - Refusal ("I don't know"): if nothing relevant is found, we say so and transfer the
      caller, instead of letting the LLM make something up (a "hallucination").

DECISION (Claude's default, review it): the chunks are turned into vectors with TF-IDF,
not with a neural embedding model. WHY: the deployed server has only 512 MB of RAM, and
sentence-transformers + PyTorch alone need more than that. Step 9 measures TF-IDF
retrieval against MiniLM embedding retrieval on a golden set so this choice is backed
by numbers.
"""

from pathlib import Path

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from app.llm import ask_llm, is_llm_configured

FAQ_DIR = Path(__file__).resolve().parent.parent / "docs" / "faq"

TOP_K = 3                    # how many chunks to retrieve for each question

# Guardrail 1 (cheap, before any LLM call): if even the best chunk scores below this
# cosine similarity, refuse. Chosen in Step 9 from the golden set: answerable questions
# scored 0.17 to 0.59, clearly off-topic ones (weather, Bitcoin) 0.00 to 0.14.
# Some unanswerable questions score higher ("mortgage from Demo Bank?" = 0.27, it shares
# the words "Demo Bank"), so a score alone cannot catch everything: that is why the LLM's
# "I don't know" (guardrail 2) exists.
MIN_RETRIEVAL_SCORE = 0.15

# Only pass chunks scoring at least 60% of the best chunk to the LLM (and list them as
# sources). This drops loosely related chunks ("How long does a transfer take?" for a
# card-delivery question) that would only distract the LLM.
RELATIVE_SCORE_CUTOFF = 0.6

REFUSAL_MESSAGE = "Sorry, I don't have that information. Let me transfer you to an agent."

# The rules for the LLM. Short answers because they are read aloud on a phone call.
SYSTEM_PROMPT = (
    "You are Demo Bank's phone assistant. Answer the customer's question using ONLY the "
    "context provided. Answer in 1 to 3 short sentences that can be read aloud on a phone "
    "call. Do not add facts, numbers or advice that are not in the context. If the context "
    "does not contain the answer, reply exactly: I don't know."
)


def load_faq_chunks(faq_dir: Path = FAQ_DIR) -> list[dict]:
    """Read every .md file and split it into one chunk per '## question' section.

    Each FAQ file looks like:
        # Card delivery
        ## How long does it take to receive my new card?
        Standard delivery ... takes 5 to 7 working days ...
    """
    chunks = []
    for path in sorted(faq_dir.glob("*.md")):
        document_text = path.read_text(encoding="utf-8")
        sections = document_text.split("\n## ")              # cut at every "## " heading
        title = sections[0].replace("# ", "").strip()        # the part before the first "## "

        for section in sections[1:]:
            lines = section.strip().splitlines()
            question = lines[0].strip()
            answer = " ".join(line.strip() for line in lines[1:]).strip()
            chunks.append({
                "source": f"{path.name}#{question}",         # where the answer came from
                "doc": path.name,
                "question": question,
                "answer": answer,
                # The text we search over: title + question + answer, so that
                # the words of the heading also help retrieval.
                "text": f"{title}. {question} {answer}",
            })
    return chunks


def build_index(chunks: list[dict]) -> tuple[TfidfVectorizer, faiss.IndexFlatIP]:
    """Turn every chunk into a TF-IDF vector and put the vectors in a FAISS index.

    For FAQ retrieval we DO remove English stopwords (unlike intent classification in
    Step 2): here we want to match the TOPIC words ("card", "delivery"), and words like
    "how" or "my" would only add noise.
    """
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])

    # FAISS needs a normal (dense) table of float32 numbers.
    vectors = matrix.toarray().astype("float32")

    # IndexFlatIP = compare by "inner product" (dot product). TF-IDF rows already have
    # length 1, so the inner product IS the cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return vectorizer, index


# Build everything ONCE when the module is imported (at API startup).
# The FAQ set is tiny, so this takes milliseconds, and we never need to keep a saved index
# file in sync with the documents: change a doc, restart, done.
CHUNKS = load_faq_chunks()
VECTORIZER, INDEX = build_index(CHUNKS)


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the top_k most similar chunks, each with a 'score' (cosine similarity 0..1)."""
    query_vector = VECTORIZER.transform([question]).toarray().astype("float32")
    scores, chunk_ids = INDEX.search(query_vector, top_k)   # both have shape (1, top_k)

    results = []
    for score, chunk_id in zip(scores[0], chunk_ids[0]):
        if chunk_id == -1:   # FAISS uses -1 when there are fewer chunks than top_k
            continue
        chunk = dict(CHUNKS[chunk_id])   # copy, so we don't change the stored chunk
        chunk["score"] = round(float(score), 4)
        results.append(chunk)
    return results


def answer_question(question: str) -> dict:
    """Answer a customer FAQ question. Never raises: on any LLM problem we fall back.

    Returns a dict with:
        answer, sources (list of "file#question"), refused (bool), used_llm (bool),
        llm_provider ("primary" = Groq, "fallback" = Ollama, "none" = no LLM used),
        top_score (float)
    """
    retrieved = retrieve(question)
    top_score = retrieved[0]["score"] if retrieved else 0.0

    # Guardrail 1: nothing relevant found -> refuse BEFORE calling the LLM (saves cost too).
    if top_score < MIN_RETRIEVAL_SCORE:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "refused": True,
                "used_llm": False, "llm_provider": "none", "top_score": top_score}

    # Only keep chunks that are relevant enough to be shown to the LLM.
    # max(...) = the stricter of the two rules (absolute minimum, or 60% of the best score).
    keep_above = max(MIN_RETRIEVAL_SCORE, RELATIVE_SCORE_CUTOFF * top_score)
    relevant = [chunk for chunk in retrieved if chunk["score"] >= keep_above]

    if is_llm_configured():
        context = "\n\n".join(
            f"[Source: {chunk['source']}]\n{chunk['question']}\n{chunk['answer']}" for chunk in relevant
        )
        try:
            # ask_llm tries Groq first and switches to local Ollama if Groq fails.
            reply = ask_llm(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=f"Context:\n{context}\n\nCustomer question: {question}\n\nAnswer:",
                max_tokens=500,   # thinking (reasoning models) + a 1-3 sentence answer
            )
            answer = reply["text"]
            # Guardrail 2: the LLM itself says the context does not contain the answer.
            # (.replace handles the "curly" apostrophe some models use.)
            if answer == "" or "i don't know" in answer.lower().replace("’", "'"):
                return {"answer": REFUSAL_MESSAGE, "sources": [], "refused": True,
                        "used_llm": True, "llm_provider": reply["provider"], "top_score": top_score}
            return {"answer": answer, "sources": [chunk["source"] for chunk in relevant],
                    "refused": False, "used_llm": True, "llm_provider": reply["provider"],
                    "top_score": top_score}
        except Exception as error:
            # Guardrail 3: BOTH LLMs failed (down, timeout, rate limit...) -> extractive answer.
            print(f"[rag] no LLM could answer ({type(error).__name__}); using the extractive fallback")

    # Extractive fallback: read out the best-matching FAQ answer word for word.
    best = relevant[0]
    return {"answer": best["answer"], "sources": [best["source"]], "refused": False,
            "used_llm": False, "llm_provider": "none", "top_score": top_score}
