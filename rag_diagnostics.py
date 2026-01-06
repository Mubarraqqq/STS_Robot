#!/usr/bin/env python3
"""
RAG System Testing and Management Utility
Test the RAG system, manage knowledge base, and verify configurations
"""

import os
import sys
import logging
from typing import List
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def check_environment():
    """Check if all required environment variables are set."""
    print_header("🔧 Checking Environment Variables")
    
    required_vars = {
        'PORCUPINE_API_KEY': 'Wake word detection API key',
        'GROQ_API_KEY': 'Groq LLM API key',
        'OPEN_API': 'OpenAI API key'
    }
    
    all_present = True
    for var, description in required_vars.items():
        value = os.getenv(var)
        status = "✅" if value else "❌"
        print(f"{status} {var:20} - {description}")
        if not value:
            all_present = False
    
    if all_present:
        print("\n✅ All environment variables are configured!")
    else:
        print("\n❌ Some environment variables are missing. Create .env file.")
    
    return all_present


def check_dependencies():
    """Check if all required packages are installed."""
    print_header("📦 Checking Dependencies")
    
    required_packages = {
        'groq': 'Groq API client',
        'speech_recognition': 'Speech recognition',
        'faiss': 'Vector search (faiss-cpu)',
        'pyttsx3': 'Text-to-speech',
        'pvporcupine': 'Wake word detection',
        'sounddevice': 'Audio input/output',
        'openai': 'OpenAI API',
        'dotenv': 'Environment variable management',
        'sklearn': 'Scikit-learn',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'langchain_text_splitters': 'LangChain text splitting'
    }
    
    missing = []
    installed = []
    
    for package, description in required_packages.items():
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package:30} - {description}")
            installed.append(package)
        except ImportError:
            print(f"❌ {package:30} - {description} (NOT INSTALLED)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True


def check_files():
    """Check if all required files exist."""
    print_header("📁 Checking Files")
    
    required_files = {
        'info.txt': 'Knowledge base (required for first run)',
        'faiss_index.idx': 'FAISS index (created on first run)',
        'embeddings.npy': 'Document embeddings (created on first run)',
        'doc_chunks.pkl': 'Text chunks (created on first run)',
        'Hey-Bruce_en_windows_v3_0_0.ppn': 'Porcupine wake word model',
        '.env': 'Environment variables'
    }
    
    critical_files = {'info.txt'}  # Must exist
    
    for filename, description in required_files.items():
        exists = os.path.exists(filename)
        status = "✅" if exists else "⚠️"
        required = " (REQUIRED)" if filename in critical_files else " (auto-generated)"
        print(f"{status} {filename:35} - {description}{required}")
    
    info_exists = os.path.exists('info.txt')
    if info_exists:
        size = os.path.getsize('info.txt') / 1024
        lines = len(open('info.txt').readlines())
        print(f"\n   📊 info.txt: {size:.1f} KB, {lines} lines")
    
    return info_exists


def test_rag_system():
    """Test the RAG system."""
    print_header("🧪 Testing RAG System")
    
    try:
        from rag_system import RAGSystem
        
        print("Initializing RAG system...")
        rag = RAGSystem()
        rag.initialize()
        print("✅ RAG system initialized\n")
        
        # Test retrieval
        test_queries = [
            "What is your name?",
            "Tell me about the system",
            "How does this work?"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            results = rag.retrieve_context(query, k=2)
            
            if results:
                for i, (chunk, score) in enumerate(results, 1):
                    confidence = rag.get_confidence_level(score)
                    chunk_preview = chunk[:80].replace('\n', ' ') + "..."
                    print(f"   [{i}] Score: {score:.3f} ({confidence}) | {chunk_preview}")
            else:
                print("   No results found")
        
        print("\n✅ RAG system test completed")
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompts():
    """Test prompt generation."""
    print_header("🎯 Testing Prompt Generation")
    
    try:
        from prompt_manager import PromptGenerator, SystemPrompt
        
        print("System Prompt Preview:")
        print("-" * 60)
        print(SystemPrompt.get_content()[:200] + "...")
        print("-" * 60 + "\n")
        
        # Test different confidence levels
        test_question = "What is the weather today?"
        test_context = "The weather is sunny and warm, with temperatures around 75°F."
        
        for confidence in ["HIGH", "MEDIUM", "LOW"]:
            print(f"\n{confidence} Confidence Prompt:")
            print("-" * 60)
            if confidence in ["HIGH", "MEDIUM"]:
                prompt = PromptGenerator.generate(
                    confidence,
                    test_question,
                    test_context
                )
            else:
                prompt = PromptGenerator.generate(
                    confidence,
                    test_question
                )
            print(prompt[:200] + "...")
            print("-" * 60)
        
        print("\n✅ Prompt generation test completed")
        return True
        
    except Exception as e:
        print(f"❌ Prompt test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speech_recognition():
    """Test speech recognition (brief test)."""
    print_header("🎤 Testing Speech Recognition")
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        
        print("Available microphones:")
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"  [{i}] {name}")
        
        print("\n⚠️  Speech recognition test requires microphone input.")
        print("Skipping live test. System will test on first use.")
        
        return True
        
    except Exception as e:
        print(f"❌ Speech recognition test failed: {e}")
        return False


def test_tts():
    """Test text-to-speech."""
    print_header("🔊 Testing Text-to-Speech")
    
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        print(f"TTS Engine: {engine.driverName}")
        print(f"Available voices: {len(engine.getProperty('voices'))}")
        
        print("\n⚠️  Skipping audio playback test.")
        print("System will test on first use.")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False


def run_diagnostics():
    """Run all diagnostic tests."""
    print("\n" + "="*60)
    print("  🔍 RAG System Diagnostics")
    print("="*60)
    
    results = {
        "Environment": check_environment(),
        "Dependencies": check_dependencies(),
        "Files": check_files(),
        "Prompts": test_prompts(),
        "Speech Recognition": test_speech_recognition(),
        "Text-to-Speech": test_tts(),
    }
    
    # Optional: Test RAG if files exist
    if results["Files"]:
        results["RAG System"] = test_rag_system()
    
    # Summary
    print_header("📋 Diagnostic Summary")
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "-"*60)
    if all_passed:
        print("✅ All diagnostics passed! Ready to run S2S_v10.py")
    else:
        print("❌ Some diagnostics failed. Please fix issues above.")
    print("-"*60 + "\n")
    
    return all_passed


def show_menu():
    """Show interactive menu."""
    while True:
        print("\n" + "="*60)
        print("  Bruce RAG System - Diagnostic & Management Tool")
        print("="*60)
        print("\n1. Run Full Diagnostics")
        print("2. Check Environment Variables")
        print("3. Check Dependencies")
        print("4. Check Files")
        print("5. Test RAG System")
        print("6. Test Prompts")
        print("7. View Knowledge Base Stats")
        print("8. Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == "1":
            run_diagnostics()
        elif choice == "2":
            check_environment()
        elif choice == "3":
            check_dependencies()
        elif choice == "4":
            check_files()
        elif choice == "5":
            test_rag_system()
        elif choice == "6":
            test_prompts()
        elif choice == "7":
            try:
                from kb_manager import KnowledgeBaseManager
                manager = KnowledgeBaseManager()
                manager.display_stats()
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "8":
            print("\nGoodbye! 👋\n")
            break
        else:
            print("❌ Invalid option. Please try again.")
        
        input("\nPress Enter to continue...")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            run_diagnostics()
        elif sys.argv[1] == "--rag":
            test_rag_system()
        elif sys.argv[1] == "--prompts":
            test_prompts()
        elif sys.argv[1] == "--interactive":
            show_menu()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python rag_diagnostics.py [--full|--rag|--prompts|--interactive]")
    else:
        show_menu()


if __name__ == "__main__":
    main()
