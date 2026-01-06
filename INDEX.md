# 📑 Complete Project Index

## 🎯 Project Overview

This is **Bruce** - a modular, production-ready Speech-to-Speech Retrieval-Augmented Generation (RAG) Assistant. The system has been refactored from a monolithic design into focused, reusable modules with comprehensive documentation.

---

## 📂 Project Structure

### 🆕 NEW MODULAR MODULES (v10)

#### 1. **rag_system.py** (407 lines)
Complete RAG pipeline implementation
- `KnowledgeBase` - Document loading and chunking
- `EmbeddingManager` - OpenAI embedding generation
- `FAISSIndex` - Vector similarity search wrapper
- `RAGRetriever` - Context retrieval and ranking
- `RAGSystem` - Main orchestrator

**Key Methods:**
```python
rag = RAGSystem()
rag.initialize()  # Load or build index
results = rag.retrieve_context(query, k=3)
confidence = rag.get_confidence_level(score)
```

#### 2. **prompt_manager.py** (286 lines)
Intelligent prompt generation system
- `PromptTemplate` - Reusable template class
- `SystemPrompt` - Assistant personality
- `PromptGenerator` - Context-aware generation
- `ConversationManager` - Chat history management
- `RAGPromptBuilder` - RAG-specific prompt construction

**Key Methods:**
```python
prompt = PromptGenerator.generate(confidence, question, context)
system = SystemPrompt.get_content()
conv = ConversationManager()
built = RAGPromptBuilder.build_rag_prompt(question, chunks, history)
```

#### 3. **kb_manager.py** (265 lines)
Knowledge base management utilities
- `KnowledgeBaseManager` - File I/O and operations
- `ContentOrganizer` - Section-based organization

**Key Methods:**
```python
manager = KnowledgeBaseManager()
manager.load_knowledge_base()
manager.append_to_knowledge_base(text)
manager.display_stats()
stats = manager.get_knowledge_base_stats()
```

#### 4. **S2S_v10.py** (433 lines)
Refactored main voice assistant
- Uses rag_system.py for RAG operations
- Uses prompt_manager.py for prompt generation
- Uses kb_manager.py optionally
- Clean, modular architecture
- Comprehensive error handling

**Key Methods:**
```python
assistant = VoiceAssistant()
assistant.run()
assistant.listen_for_wake_word()
assistant.start_conversation()
assistant.speak(text)
```

#### 5. **rag_diagnostics.py** (385 lines)
Testing and diagnostic suite
- Environment variable validation
- Dependency checking
- File verification
- RAG system testing
- Prompt testing
- Interactive menu

**Usage:**
```bash
python rag_diagnostics.py --full           # Full diagnostics
python rag_diagnostics.py --interactive    # Interactive menu
python rag_diagnostics.py --rag            # Test RAG
python rag_diagnostics.py --prompts        # Test prompts
```

---

### 📚 DOCUMENTATION

#### 1. **MODULAR_GUIDE.md** (Comprehensive)
Complete usage guide covering:
- Feature overview
- Architecture explanation
- Setup instructions
- Customization guide
- API reference
- Troubleshooting
- Best practices

**When to read:** Start here for detailed understanding

#### 2. **QUICK_REFERENCE.md** (Quick Lookup)
Fast reference guide for:
- File structure
- Common tasks
- Configuration options
- Troubleshooting table
- Module dependencies
- Performance tips

**When to read:** Need quick answers

#### 3. **IMPROVEMENTS_SUMMARY.md** (What Changed)
Detailed explanation of:
- What was refactored
- Benefits of each module
- Before/after comparison
- Migration guide
- Key improvements

**When to read:** Understand what's new

#### 4. **IMPROVEMENTS.txt** (Quick Overview)
Summary of all improvements:
- Modularization details
- Prompt improvements
- Knowledge base features
- System tools
- Getting started

**When to read:** Quick overview of changes

---

### 📄 CONFIGURATION & DATA

#### 1. **.env** (YOUR CONFIGURATION - CREATE THIS)
Required API keys:
```env
PORCUPINE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPEN_API=your_key_here
```

#### 2. **info.txt** (KNOWLEDGE BASE)
Plain text file with your knowledge content
- Already populated with system information
- Edit to add your custom knowledge
- Optional section-based organization

#### 3. **requirements.txt** (DEPENDENCIES)
Python package requirements:
```
groq
SpeechRecognition
faiss-cpu
python-dotenv
scikit-learn
...and more
```

#### 4. **hey-bruce_*.ppn** (WAKE WORD MODEL)
Porcupine wake word detection model
- Detects "Hey Bruce" phrase
- Pre-trained model included

---

### 🗂️ AUTO-GENERATED FILES (First Run)

These are created automatically on first run:
- **faiss_index.idx** - Vector similarity index
- **embeddings.npy** - Document embeddings
- **doc_chunks.pkl** - Text chunks cache

No action needed - system creates these.

---

### 📅 LEGACY FILES (Reference Only)

- **S2S_v8.py** - Original monolithic version
- **S2S_v9.py** - Previous version
- **README.md** - Original documentation

*These are preserved for reference. Use S2S_v10.py instead.*

---

## 🚀 Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
cat > .env << EOF
PORCUPINE_API_KEY=your_key
GROQ_API_KEY=your_key
OPEN_API=your_key
EOF
```

### 2. Verify
```bash
python rag_diagnostics.py --full
```

### 3. Customize (Optional)
```bash
# Edit knowledge base
nano info.txt

# Edit prompts
nano prompt_manager.py
```

### 4. Run
```bash
python S2S_v10.py
```

### 5. Interact
```
Say: "Hey Bruce!"
Ask: "Your question"
Say: "Goodbye" to exit
```

---

## 📋 File Dependencies

```
S2S_v10.py (main application)
├── imports rag_system.py
│   ├── imports numpy, faiss, openai
│   ├── imports langchain_text_splitters
│   └── imports sklearn
├── imports prompt_manager.py
│   └── minimal dependencies
└── (optional) imports kb_manager.py

rag_diagnostics.py (testing tool)
├── imports rag_system.py
├── imports prompt_manager.py
└── imports kb_manager.py

kb_manager.py (standalone utility)
└── minimal dependencies
```

---

## 🔧 Configuration Guide

### Environment Variables
Located in: `.env`
```env
PORCUPINE_API_KEY    # From picovoice.ai
GROQ_API_KEY         # From console.groq.com
OPEN_API             # From platform.openai.com
```

### Speech Recognition Settings
Located in: `S2S_v10.py` → `_configure_speech_recognition()`
- `energy_threshold` - Noise sensitivity
- `pause_threshold` - Pause detection
- `phrase_threshold` - Minimum audio length

### Text-to-Speech Settings
Located in: `S2S_v10.py` → `_configure_tts()`
- `rate` - Speaking speed (100-200)
- `volume` - Volume level (0.0-1.0)

### RAG Parameters
Located in: `rag_system.py`
- `chunk_size` - Document chunk size (default: 500)
- `chunk_overlap` - Overlap between chunks (default: 50)
- `k` - Number of chunks to retrieve (default: 5)
- Confidence thresholds: 0.85 (HIGH), 0.75 (MEDIUM)

### Prompt Templates
Located in: `prompt_manager.py` → `PromptGenerator.TEMPLATES`
- Edit text for HIGH confidence level
- Edit text for MEDIUM confidence level
- Edit text for LOW confidence level

### System Personality
Located in: `prompt_manager.py` → `SystemPrompt.SYSTEM_CONTENT`
- Update personality description
- Change tone and style
- Modify behavior guidelines

---

## 🧪 Testing Guide

### Full System Test
```bash
python rag_diagnostics.py --full
```
Checks everything and provides detailed report.

### Component Testing
```bash
# Test RAG retrieval
python rag_diagnostics.py --rag

# Test prompt generation
python rag_diagnostics.py --prompts

# Interactive menu
python rag_diagnostics.py --interactive
```

### Manual Module Testing
```python
# Test RAG
from rag_system import RAGSystem
rag = RAGSystem()
rag.initialize()
results = rag.retrieve_context("test", k=3)

# Test prompts
from prompt_manager import PromptGenerator
prompt = PromptGenerator.generate("HIGH", "Q", "context")

# Test KB manager
from kb_manager import KnowledgeBaseManager
mgr = KnowledgeBaseManager()
mgr.display_stats()
```

---

## 📊 System Architecture

### Data Flow
```
User Voice Input
    ↓ [S2S_v10.py]
Speech Recognition (Google STT)
    ↓ [prompt_manager.py]
Text Preprocessing
    ↓ [rag_system.py]
Embedding Generation (OpenAI)
    ↓
FAISS Vector Search
    ↓
Retrieve Context
    ↓ [prompt_manager.py]
Confidence Analysis
    ↓
Prompt Generation
    ↓ [S2S_v10.py]
LLM API Call (Groq)
    ↓
Text Response
    ↓
Text-to-Speech (pyttsx3)
    ↓
Voice Output
```

### Module Interaction
```
rag_system.py
  ├─ KnowledgeBase (load/chunk documents)
  ├─ EmbeddingManager (create embeddings)
  ├─ FAISSIndex (vector search)
  └─ RAGRetriever (get relevant chunks)

prompt_manager.py
  ├─ SystemPrompt (define personality)
  ├─ PromptTemplate (reusable formats)
  ├─ PromptGenerator (create prompts)
  ├─ ConversationManager (track history)
  └─ RAGPromptBuilder (assemble final prompt)

kb_manager.py
  ├─ KnowledgeBaseManager (file operations)
  └─ ContentOrganizer (organize sections)

S2S_v10.py (main application)
  └─ Uses all modules for voice interaction
```

---

## 🎯 Use Cases

### 1. Basic Usage (Out of the Box)
- Wake word detection
- Question answering from knowledge base
- Time/date queries
- Conversation management

### 2. Customization
- Edit prompts for specific domain
- Add custom knowledge to info.txt
- Adjust confidence thresholds
- Change assistant personality

### 3. Integration
- Import modules into other projects
- Use RAG system as library
- Create custom retrievers
- Build web interfaces

### 4. Extension
- Add new prompt templates
- Implement re-ranking strategies
- Integrate external APIs
- Create specialized knowledge bases

---

## 🔒 Security Considerations

### Secrets Management
- Never commit `.env` file
- Use environment variables for keys
- Regenerate keys if exposed
- Use version control exclusions

### Data Privacy
- Local processing where possible
- API-based operations for embeddings/LLM
- No persistent conversation storage (default)
- Consider implementing logging guards

### Best Practices
- Validate user input
- Use HTTPS for any web integrations
- Monitor API usage
- Keep dependencies updated

---

## 📈 Performance Optimization

### First Run
- Takes 1-2 minutes for embedding generation
- Creates FAISS index
- Caches embeddings

### Subsequent Runs
- 2-3 second startup (loads cache)
- Fast RAG retrieval (~100ms)
- LLM response time depends on API

### Optimization Tips
- Keep knowledge base focused
- Use good document structure
- Monitor similarity score distribution
- Consider batch embedding for large KBs

### Cost Management
- OpenAI embeddings: ~$0.02/1M tokens
- Groq: Free tier available
- Monitor API usage regularly
- Cache embeddings to reduce calls

---

## 🐛 Troubleshooting

### Common Issues
| Issue | Solution |
|-------|----------|
| Missing API keys | Create `.env` with required keys |
| Module not found | Run `pip install -r requirements.txt` |
| Microphone error | Check microphone in system settings |
| No context retrieved | Check similarity scores in logs |
| Slow responses | Verify API keys and internet |

### Debug Steps
1. Run diagnostics: `python rag_diagnostics.py --full`
2. Check console logs for error messages
3. Test individual modules independently
4. Review configuration in respective files
5. Check knowledge base content and format

### Logging
- System logs to console with timestamps
- Enable debug logging by modifying `logging.basicConfig`
- Check logs for specific error information

---

## 📚 Learning Path

### For Beginners
1. Read `QUICK_REFERENCE.md` (overview)
2. Read `IMPROVEMENTS.txt` (what's new)
3. Run `python rag_diagnostics.py --full`
4. Try running assistant: `python S2S_v10.py`

### For Developers
1. Read `MODULAR_GUIDE.md` (complete guide)
2. Review source code with docstrings
3. Study `rag_system.py` (RAG implementation)
4. Review `prompt_manager.py` (prompt logic)
5. Examine `S2S_v10.py` (integration)

### For Customizers
1. Understand module architecture
2. Edit `prompt_manager.py` for custom prompts
3. Update `info.txt` with domain knowledge
4. Test with `rag_diagnostics.py`
5. Deploy customized version

---

## 🚀 Next Steps

1. **Verify Setup**
   ```bash
   python rag_diagnostics.py --full
   ```

2. **Customize Content**
   - Edit `info.txt` with your knowledge
   - Edit `prompt_manager.py` for your style

3. **Test System**
   ```bash
   python S2S_v10.py
   ```

4. **Interact**
   - Say "Hey Bruce"
   - Ask questions
   - Get smart answers

5. **Monitor & Improve**
   - Check logs for errors
   - Review RAG confidence scores
   - Refine knowledge base

---

## 📞 Support Resources

### Documentation
- **MODULAR_GUIDE.md** - Comprehensive guide
- **QUICK_REFERENCE.md** - Quick lookup
- **Source code docstrings** - Implementation details

### Testing
- **rag_diagnostics.py** - System health checks
- **Module imports** - Individual testing
- **Console logs** - Detailed information

### Code Quality
- Clear module separation
- Comprehensive docstrings
- Type hints throughout
- Error handling and logging

---

## ✅ Verification Checklist

- [ ] All Python files created
- [ ] Diagnostics pass: `python rag_diagnostics.py --full`
- [ ] `.env` file with API keys created
- [ ] `info.txt` populated with knowledge
- [ ] Can run: `python S2S_v10.py`
- [ ] Say "Hey Bruce" activates system
- [ ] Can ask questions and get responses
- [ ] Logs show reasonable confidence scores

---

## 🎉 Summary

Your RAG system is now:
- ✅ Modularized into focused components
- ✅ Better prompts with confidence levels
- ✅ Improved knowledge base management
- ✅ Comprehensive diagnostics
- ✅ Extensively documented
- ✅ Production-ready

**Start with:** `python S2S_v10.py`  
**Questions?** Read: `MODULAR_GUIDE.md`  
**Quick lookup?** See: `QUICK_REFERENCE.md`

---

**Your intelligent voice assistant is ready to go! 🚀**
