# 🚀 Quick Reference Guide

## File Structure Overview

```
STS_Robot/
├── 📄 Core Application Files (NEW)
│   ├── S2S_v10.py              ← Main assistant (REFACTORED - use this!)
│   ├── rag_system.py           ← RAG pipeline (NEW)
│   ├── prompt_manager.py       ← Prompt generation (NEW)
│   ├── kb_manager.py           ← Knowledge base utilities (NEW)
│   └── rag_diagnostics.py      ← Testing & diagnostics (NEW)
│
├── 📄 Legacy Files (Reference Only)
│   ├── S2S_v8.py               ← Original monolithic version
│   ├── S2S_v9.py               ← Previous version
│   └── README.md               ← Original documentation
│
├── 📚 Knowledge Base & Indexes
│   ├── info.txt                ← Your knowledge base content
│   ├── faiss_index.idx         ← Vector index (auto-generated)
│   ├── embeddings.npy          ← Document embeddings (auto-generated)
│   └── doc_chunks.pkl          ← Text chunks (auto-generated)
│
├── 🔑 Configuration
│   ├── .env                    ← API keys (create this!)
│   ├── requirements.txt        ← Python dependencies
│   └── Hey-Bruce_*.ppn         ← Wake word model
│
└── 📖 Documentation (NEW)
    ├── IMPROVEMENTS_SUMMARY.md ← What was changed
    ├── MODULAR_GUIDE.md        ← Complete usage guide
    └── this file               ← Quick reference
```

---

## 🎯 Quick Start (3 Steps)

### Step 1: Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env with API keys
cat > .env << EOF
PORCUPINE_API_KEY=your_porcupine_key
GROQ_API_KEY=your_groq_key
OPEN_API=your_openai_key
EOF
```

### Step 2: Prepare Knowledge Base
```bash
# Create or update info.txt
echo "Your knowledge content here..." > info.txt
```

### Step 3: Run
```bash
# Run diagnostics (optional but recommended)
python rag_diagnostics.py --full

# Start the assistant
python S2S_v10.py
```

---

## 📋 Common Tasks

### Check Everything is Working
```bash
python rag_diagnostics.py --full
```

### Run Interactive Menu
```bash
python rag_diagnostics.py --interactive
```

### Update Knowledge Base
```python
from kb_manager import KnowledgeBaseManager

manager = KnowledgeBaseManager()
manager.append_to_knowledge_base("New information...")
```

### View Knowledge Base Stats
```python
from kb_manager import KnowledgeBaseManager

manager = KnowledgeBaseManager()
manager.display_stats()
```

### Test RAG Retrieval
```python
from rag_system import RAGSystem

rag = RAGSystem()
rag.initialize()
results = rag.retrieve_context("your question", k=3)
for chunk, score in results:
    print(f"{score:.3f}: {chunk[:100]}")
```

### Test Prompt Generation
```python
from prompt_manager import PromptGenerator

prompt = PromptGenerator.generate(
    "HIGH",
    "What is X?",
    "Context about X..."
)
print(prompt)
```

### Modify System Personality
Edit `prompt_manager.py`, find `SystemPrompt.SYSTEM_CONTENT` and update the text.

### Customize Response Prompts
Edit `prompt_manager.py`, find `PromptGenerator.TEMPLATES` and modify the templates.

---

## 🔧 Configuration

### Environment Variables (.env)
```env
PORCUPINE_API_KEY=sk_...         # From picovoice.ai
GROQ_API_KEY=gsk_...              # From console.groq.com
OPEN_API=sk-...                   # From platform.openai.com
```

### Knowledge Base (info.txt)
Plain text file with your content. Optional structure:
```
--- Section Name ---
Your content here...

--- Another Section ---
More content...
```

### Speech Recognition Settings
In `S2S_v10.py` → `_configure_speech_recognition()`:
- `energy_threshold` - Sensitivity to background noise
- `pause_threshold` - How long to wait for silence
- `non_speaking_duration` - How long speech can be silent

### TTS Settings
In `S2S_v10.py` → `_configure_tts()`:
- `rate` - Speaking speed (100-200 typical)
- `volume` - Volume level (0.0-1.0)

### RAG Parameters
In `rag_system.py` → `RAGRetriever.retrieve()`:
- `k` - Number of chunks to retrieve (default: 5)
- Similarity thresholds in `RAGPromptBuilder.build_rag_prompt()`

---

## 🧪 Testing

### Full Diagnostic Suite
```bash
python rag_diagnostics.py --full
```
Checks: environment, dependencies, files, RAG system, prompts

### Individual Component Tests
```bash
# Test RAG only
python rag_diagnostics.py --rag

# Test prompts only
python rag_diagnostics.py --prompts

# Interactive menu
python rag_diagnostics.py --interactive
```

### Manual Testing
```python
# Test specific module
from rag_system import RAGSystem
from prompt_manager import PromptGenerator
from kb_manager import KnowledgeBaseManager

# Initialize and test each
```

---

## 📊 Understanding Confidence Levels

```
Similarity Score → Confidence → Strategy
≥0.85           → HIGH       → Use 2 best chunks
≥0.75           → MEDIUM     → Use 1 chunk + acknowledge
<0.75           → LOW        → General knowledge only
```

Example flow:
1. User asks question
2. System finds most similar document chunk
3. Calculates similarity score (0-1)
4. Determines confidence level
5. Generates appropriate prompt
6. Gets LLM response
7. Speaks answer

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Missing environment variables" | Create `.env` file with API keys |
| "Module not found" | Run `pip install -r requirements.txt` |
| "Could not understand audio" | Check microphone, reduce background noise |
| "No knowledge base found" | Create `info.txt` with content |
| "RAG not retrieving context" | Check similarity scores in logs |
| "Slow responses" | Verify API keys work, check internet |

---

## 🔄 Module Dependencies

```
S2S_v10.py (main)
├── imports rag_system.py
├── imports prompt_manager.py
└── imports kb_manager.py (optional)

rag_system.py (standalone)
└── imports: openai, faiss, numpy, sklearn

prompt_manager.py (standalone)
└── no heavy dependencies

kb_manager.py (standalone)
└── imports: pickle, logging

rag_diagnostics.py (testing only)
├── imports rag_system.py
├── imports prompt_manager.py
└── imports kb_manager.py
```

---

## 📈 Performance Tips

1. **First Run**: Takes 1-2 minutes (embedding generation)
2. **Subsequent Runs**: Uses cached index (2-3 seconds startup)
3. **Faster Responses**: Ensure good internet for API calls
4. **Better Retrieval**: Organize knowledge base by topics
5. **Lower Costs**: Keep knowledge base focused

---

## 🎓 Architecture Overview

```
User Voice Input
    ↓
[Speech Recognition] ← S2S_v10.py
    ↓
Text Transcription
    ↓
[RAG System] ← rag_system.py
├─ Embed query
├─ Search FAISS index
└─ Retrieve context
    ↓
[Prompt Generation] ← prompt_manager.py
├─ Check confidence level
├─ Select appropriate prompt
└─ Build final prompt
    ↓
[LLM API] (Groq)
    ↓
Text Response
    ↓
[Text-to-Speech] ← S2S_v10.py
    ↓
Voice Output
```

---

## 🚨 Important Notes

### Security
- Never commit `.env` file
- Regenerate API keys if exposed
- Keep local copies of embeddings

### Cost Management
- OpenAI: ~$0.02 per 1M tokens (embeddings)
- Groq: Free tier available
- Monitor API usage

### Performance
- FAISS index kept in memory
- Chunks cached as pickle
- Embeddings cached as numpy array

### Customization
- Prompts: Edit `prompt_manager.py`
- Knowledge base: Edit `info.txt` or use `kb_manager.py`
- Personality: Update `SystemPrompt`
- RAG behavior: Modify `rag_system.py` thresholds

---

## 📚 File Reference

### rag_system.py
```python
RAGSystem()                    # Main class
  .initialize()                # Load or build index
  .retrieve_context(query)     # Get relevant chunks
  .get_confidence_level(score) # Determine confidence
```

### prompt_manager.py
```python
SystemPrompt.get_content()          # Get system prompt
PromptGenerator.generate(level, q, c) # Generate prompt
ConversationManager(max_history)    # Manage chat history
RAGPromptBuilder.build_rag_prompt() # Build complete prompt
```

### kb_manager.py
```python
KnowledgeBaseManager()          # Main class
  .load_knowledge_base()        # Read content
  .save_knowledge_base(content) # Write content
  .append_to_knowledge_base()   # Add content
  .get_knowledge_base_stats()   # Get statistics
  .display_stats()              # Print stats
```

### S2S_v10.py
```python
VoiceAssistant()                # Main class
  .run()                         # Start assistant
  .listen_for_wake_word()        # Listen for "Hey Bruce"
  .start_conversation()          # Begin chat
  .listen_for_command()          # Get user input
  .get_ai_response(input)        # Get LLM response
  .speak(text)                   # Output audio
```

---

## 🎯 Next Steps

1. **Verify Setup**: `python rag_diagnostics.py --full`
2. **Customize Prompts**: Edit `prompt_manager.py`
3. **Update Knowledge Base**: Edit `info.txt`
4. **Run Assistant**: `python S2S_v10.py`
5. **Say**: "Hey Bruce!"

---

## 📞 Need Help?

1. **Check logs** - Console shows detailed information
2. **Run diagnostics** - `python rag_diagnostics.py --full`
3. **Read guides** - `MODULAR_GUIDE.md` has detailed docs
4. **Review source** - Code has docstrings and comments
5. **Test modules** - Each module can be tested independently

---

**Ready to use your improved RAG system!** 🚀
