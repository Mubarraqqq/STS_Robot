# 🗣️ Modularized Speech-to-Speech RAG Assistant (Bruce)

A highly modular, production-ready voice assistant that combines speech recognition, Retrieval-Augmented Generation (RAG), and natural language processing for intelligent conversations.

## 🎯 What's New (v10)

### ✅ Modularization
The monolithic v8/v9 codebase has been refactored into focused, reusable modules:

- **`rag_system.py`** - Complete RAG pipeline
  - `KnowledgeBase` - Document management
  - `EmbeddingManager` - OpenAI embeddings
  - `FAISSIndex` - Vector similarity search
  - `RAGRetriever` - Context retrieval
  - `RAGSystem` - Main orchestrator

- **`prompt_manager.py`** - Intelligent prompt generation
  - `SystemPrompt` - Core assistant personality
  - `PromptTemplate` - Reusable prompt templates
  - `PromptGenerator` - Context-aware prompt generation
  - `ConversationManager` - Chat history management
  - `RAGPromptBuilder` - RAG-specific prompt construction

- **`kb_manager.py`** - Knowledge base utilities
  - `KnowledgeBaseManager` - File and chunk management
  - `ContentOrganizer` - Section-based organization

- **`S2S_v10.py`** - Refactored voice assistant
  - Clean, modular design
  - Dependency injection
  - Better error handling

---

## 🚀 Features

### 🎙️ **Voice Interaction**
- Wake word detection ("Hey Bruce") via Porcupine
- Real-time speech recognition (Google Speech-to-Text)
- Natural text-to-speech output (pyttsx3)

### 🧠 **Advanced RAG System**
- **Vector-based retrieval**: FAISS for fast similarity search
- **Smart confidence levels**: HIGH (≥0.85), MEDIUM (≥0.75), LOW (<0.75)
- **Context-aware responses**: Different prompts for different confidence levels
- **Fallback to general knowledge**: Seamless degradation when no relevant docs found

### 💬 **Intelligent Conversation**
- **Conversation memory**: Maintains context across exchanges
- **Special commands**: Time, date, day queries answered instantly
- **Conversation control**: Natural goodbye/exit detection
- **Timeout management**: 60-second idle timeout before disconnect

### 🔧 **Production-Ready Code**
- Comprehensive logging
- Error handling and recovery
- Resource cleanup
- Configuration management

---

## 📁 Project Structure

```
STS_Robot/
├── S2S_v10.py                    # 🆕 Main modular assistant
├── rag_system.py                 # 🆕 Complete RAG pipeline
├── prompt_manager.py             # 🆕 Prompt generation system
├── kb_manager.py                 # 🆕 Knowledge base utilities
│
├── S2S_v8.py / S2S_v9.py        # Legacy versions (reference)
├── .env                          # Environment variables (API keys)
├── requirements.txt              # Python dependencies
│
├── info.txt                      # 📚 Knowledge base (your content)
├── embeddings.npy                # Pre-computed document embeddings
├── faiss_index.idx               # FAISS vector index
├── doc_chunks.pkl                # Pickled document chunks
│
├── Hey-Bruce_en_windows_v3_0_0.ppn  # Porcupine wake word model
└── README.md                     # This file
```

---

## 🛠️ Setup Instructions

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Configure Environment**

Create a `.env` file in the project directory:

```env
PORCUPINE_API_KEY=your_porcupine_api_key_here
GROQ_API_KEY=your_groq_api_key_here
OPEN_API=your_openai_api_key_here
```

**Get API Keys:**
- [Porcupine](https://console.picovoice.ai/) - Wake word detection
- [Groq](https://console.groq.com/) - Fast LLM inference
- [OpenAI](https://platform.openai.com/api-keys) - Embeddings

### 3. **Prepare Knowledge Base**

Create or update `info.txt` with your content:

```
Your knowledge base content here...
Topics, facts, information that Bruce should know about.
```

Optionally organize with sections:

```
--- History ---
Your history content...

--- Technology ---
Your tech content...

--- General ---
Other information...
```

### 4. **Run the Assistant**

```bash
python S2S_v10.py
```

The assistant will:
1. Load or create the RAG index on first run
2. Listen for "Hey Bruce"
3. Start conversation when wake word detected
4. Continue until you say goodbye or timeout occurs

---

## 🧬 How the RAG System Works

### **Retrieval Pipeline**

```
User Question
    ↓
Generate Embedding (OpenAI)
    ↓
FAISS Similarity Search (k=3)
    ↓
Calculate Confidence Score
    ↓
Select Prompt Strategy (HIGH/MEDIUM/LOW)
    ↓
Build Context-Aware Prompt
    ↓
LLM Response (Groq Gemma2-9B)
    ↓
Add to Conversation History
    ↓
Text-to-Speech Output
```

### **Confidence Levels**

| Confidence | Threshold | Strategy | Use Case |
|-----------|-----------|----------|----------|
| **HIGH** | ≥0.85 | Use top 2 relevant chunks | Factual queries with good matches |
| **MEDIUM** | ≥0.75 | Use top chunk with uncertainty note | Partial matches |
| **LOW** | <0.75 | General knowledge only | Novel questions |

---

## 💡 Usage Examples

### **Starting a Conversation**

```
You: "Hey Bruce!"
Bruce: "Hi, I'm Bruce. How can I help you today?"
You: "What time is it?"
Bruce: "The current time is 02:30 PM on Monday, January 6, 2025."
```

### **Knowledge-Based Query**

```
You: "Tell me about [topic from your knowledge base]"
Bruce: [Retrieves relevant information and provides answer]
```

### **Fallback to General Knowledge**

```
You: "What's 2+2?"
Bruce: [Uses general knowledge since no match in KB]
```

### **Ending Conversation**

```
You: "Goodbye"
Bruce: "Goodbye! Say 'Hey Bruce' if you need me again."
```

---

## 🔄 Customizing the System

### **Improve Prompts**

Edit `prompt_manager.py` → `PromptGenerator.TEMPLATES`:

```python
TEMPLATES = {
    "HIGH": PromptTemplate(
        "high_confidence",
        """Your custom prompt here..."""
    ),
    # ... more templates
}
```

### **Adjust System Personality**

Edit `prompt_manager.py` → `SystemPrompt.SYSTEM_CONTENT`:

```python
SYSTEM_CONTENT = """You are [your custom personality]..."""
```

### **Update Knowledge Base**

Option 1: Direct file edit
```bash
vim info.txt  # Edit directly
```

Option 2: Programmatic update
```python
from kb_manager import KnowledgeBaseManager

manager = KnowledgeBaseManager()
manager.append_to_knowledge_base("New information here...")
```

### **Fine-tune RAG Parameters**

In `S2S_v10.py` → `_initialize_rag()`:

```python
self.rag_system = RAGSystem(
    kb_path="info.txt",  # Change if needed
    faiss_path="faiss_index.idx",
)
```

Or in `rag_system.py` → `RAGRetriever.retrieve()`:

```python
def retrieve(self, query_embedding, k=5):  # Change k to retrieve more/fewer chunks
    ...
```

---

## 📊 Monitoring & Diagnostics

### **View Knowledge Base Stats**

```python
from kb_manager import KnowledgeBaseManager

manager = KnowledgeBaseManager()
manager.display_stats()
```

**Output:**
```
==================================================
📊 Knowledge Base Statistics
==================================================
Exists: True
Size: 25.34 KB
Characters: 25,948
Words: 4,287
Lines: 182
==================================================
```

### **Check RAG System Health**

```python
from rag_system import RAGSystem

rag = RAGSystem()
rag.initialize()
results = rag.retrieve_context("test query", k=3)
for chunk, score in results:
    print(f"Score: {score:.3f} | {chunk[:100]}...")
```

### **Logs**

The system logs to console with timestamps:

```
2025-01-06 14:30:45 - INFO - ✅ All required environment variables present
2025-01-06 14:30:46 - INFO - ✅ All components initialized successfully
2025-01-06 14:30:47 - INFO - 👂 Voice Assistant started. Listening for 'Hey Bruce'...
```

---

## 🔒 Best Practices

### **Security**
- Never commit `.env` file to version control
- Use strong API keys from official providers
- Regenerate keys if exposed

### **Performance**
- First run will take longer (RAG indexing)
- Subsequent runs use cached index
- Keep knowledge base organized for better retrieval
- Monitor API usage for cost

### **Maintenance**
- Regularly update knowledge base
- Test new prompts before deployment
- Review logs for errors
- Keep dependencies updated

---

## 🐛 Troubleshooting

### **"Missing environment variables"**
- Create `.env` file with required keys
- Verify key names match exactly

### **"Could not understand audio"**
- Ensure microphone is working
- Reduce background noise
- Check `recognizer` settings in `S2S_v10.py`

### **Slow responses**
- Check internet connection (API calls)
- Verify GPU availability if using local embeddings
- Review knowledge base size

### **RAG not retrieving context**
- Verify `info.txt` exists and has content
- Check that embeddings were generated
- Review similarity scores in logs

---

## 📚 API Reference

### **RAGSystem**

```python
from rag_system import RAGSystem

rag = RAGSystem()
rag.initialize()
results = rag.retrieve_context(query="your question", k=3)
confidence = rag.get_confidence_level(score=0.82)
```

### **PromptGenerator**

```python
from prompt_manager import PromptGenerator

prompt = PromptGenerator.generate(
    confidence_level="HIGH",
    question="What is X?",
    context="Answer: X is..."
)
```

### **ConversationManager**

```python
from prompt_manager import ConversationManager

conv = ConversationManager(max_history=4)
conv.add_message("user", "Hello")
conv.add_message("assistant", "Hi there!")
history = conv.get_history()
conv.clear()
```

---

## 🚀 Future Enhancements

- [ ] Multi-language support
- [ ] Local embedding model (eliminate OpenAI dependency)
- [ ] Web UI dashboard
- [ ] Persistent conversation logging
- [ ] User-specific knowledge bases
- [ ] Advanced RAG with re-ranking
- [ ] Voice activity detection
- [ ] Custom wake word training

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

- **Picovoice Porcupine**: Wake word detection
- **OpenAI**: Embeddings and API
- **Groq**: Fast LLM inference
- **FAISS**: Vector similarity search
- **LangChain**: Text splitting utilities

---

## 💬 Support

For issues, questions, or suggestions:
1. Check the Troubleshooting section
2. Review logs for error messages
3. Verify environment configuration
4. Test individual components in isolation

---

**Made with ❤️ - Your Intelligent Voice Assistant**
