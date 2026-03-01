## Implementation Plan for Week 4: Textual Data Agent (Starting March 2, 2026)

**Overall Goal for the Week**: Prototype a basic working agent that collects text from sources (NSE news, Economic Times, RBI reports, earnings transcripts, X posts), handles PDF extraction and Hinglish, and outputs clean data for sentiment (Phase 2). Use provenance tagging and quality checks. By end of week, have it ready for partner feedback/meeting.

### Day 1: March 2 (Sunday) - Setup & Planning (Light Start)
- Review your uploaded docs (`plan1.3.7.docx`, notes on Textual Agent).
- In Antigravity: Create a new file `textual_data_agent.py` and prompt for a basic class skeleton (e.g., "Build a Python class for Textual Data Agent with methods for source collection, PDF extraction, X search").
- Research quick: Browse NSE/Economic Times APIs/docs (use tool if needed, but start with known libraries like `requests`, `tweepy` for X).
- Output: Basic code outline + list of libraries needed (e.g., `feedparser` for RSS, `PyPDF2` for extraction).
- Partner check: Share outline via chat.

### Day 2: March 3 (Monday) - Source Collection Basics
- Focus on core sources: NSE news, Economic Times, RBI reports.
- Prompt Antigravity: "Implement fetch methods for NSE news RSS and Economic Times articles; add provenance tagging (`source_type`, `ingestion_timestamp`)".
- Add simple storage: Save fetched text as JSON with timestamps.
- Test: Run locally, check 5-10 items.
- Output: Working fetch code for 2-3 sources.
- Quick win: Note how it ties to sentiment (e.g., RBI report = high-impact).

### Day 3: March 4 (Tuesday) - X Social Data & Keywords
- Add X (Twitter) collection: Use documented keyword/semantic search rules (e.g., "RBI rate hike" OR "Nifty crash").
- Prompt: "Add X fetch using tweepy with keyword/semantic rules; handle rate limits and provenance".
- Filter for India-relevant (e.g., `#NSE`, `#RBI`).
- Test: Pull sample posts, save with tags.
- Output: Integrated X method; total sources now 4+.
- Pro tip: Use your Pune location for any geo-filters if relevant.

### Day 4: March 5 (Wednesday) - PDF Extraction & Hinglish Handling
- Handle PDFs (RBI reports, earnings transcripts): Implement extraction pipeline.
- Prompt: "Build PDF extraction with PyPDF2 or pdfplumber; add spot checks for accuracy".
- For code-mixed English/Hinglish: Document strategy (e.g., transliterate with `indic-nlp-library`, normalize slang).
- Prompt: "Add Hinglish handling: detect/transliterate code-mixed text for sentiment readiness".
- Test: Upload a sample RBI PDF, extract + clean.
- Output: Full pipeline with validation (spot checks).

### Day 5: March 6 (Thursday) - Quality Checks & Integration
- Add data quality: Check for missing text, duplicates, spam (e.g., simple length/filter rules).
- Prompt: "Implement quality checks: missingness, outliers (too short/long), and provenance for all sources".
- Integrate: Make agent output clean JSON/Parquet for Silver layer (with `quality_status`: `pass`/`warn`/`fail`).
- Test end-to-end: Run full agent, simulate input to Sentiment Agent.
- Output: Complete prototype code.

### Day 6: March 7 (Friday) - Testing & Bug Fixes
- Run tests: Leakage checks (time alignment), spot checks on PDFs/X.
- Prompt Antigravity for unit tests: "Add pytest tests for fetch, extraction, Hinglish handling".
- Fix bugs: Debug any failures (e.g., API rate limits).
- Add logging: For provenance, failures.
- Output: Tested, debugged agent ready for review.
- Partner sync: Share code/repo link.

### Day 7: March 8 (Saturday) - Review & Polish
- Self-review: Run on real data (e.g., recent RBI bulletin + X posts).
- Document: Add comments, README snippet for the agent.
- Prep for meeting: Note improvements (e.g., "Added Hinglish for better Indian sentiment").
- Optional: Quick prompt for optimizations (e.g., "Make it faster with async fetches").
- Output: Polished code + notes for partner discussion.
- Relax: You've built a key Phase 1 piece - celebrate with chai!

This keeps you on track with your ramp-up (understanding -> code -> test). Adjust for your hours/energy, and hit me if a day feels stuck!
