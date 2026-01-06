"""
Knowledge Base Manager
Utilities for managing and updating the knowledge base
"""

import os
import logging
from typing import List, Optional
import pickle

logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """Manage the knowledge base file and chunks."""
    
    def __init__(self, kb_path: str = "info.txt", chunks_path: str = "doc_chunks.pkl"):
        """
        Initialize knowledge base manager.
        
        Args:
            kb_path: Path to knowledge base text file
            chunks_path: Path to chunks pickle file
        """
        self.kb_path = kb_path
        self.chunks_path = chunks_path
    
    def load_knowledge_base(self) -> str:
        """
        Load the knowledge base text.
        
        Returns:
            Knowledge base content as string
        """
        try:
            if not os.path.exists(self.kb_path):
                logger.warning(f"Knowledge base not found at {self.kb_path}")
                return ""
            
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"✅ Loaded {len(content)} characters from knowledge base")
            return content
            
        except Exception as e:
            logger.error(f"❌ Error loading knowledge base: {e}")
            return ""
    
    def save_knowledge_base(self, content: str) -> bool:
        """
        Save content to knowledge base file.
        
        Args:
            content: Text content to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.kb_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Saved {len(content)} characters to knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving knowledge base: {e}")
            return False
    
    def append_to_knowledge_base(self, content: str) -> bool:
        """
        Append content to existing knowledge base.
        
        Args:
            content: Text to append
            
        Returns:
            True if successful, False otherwise
        """
        try:
            existing = self.load_knowledge_base()
            updated = existing + "\n\n" + content if existing else content
            return self.save_knowledge_base(updated)
            
        except Exception as e:
            logger.error(f"❌ Error appending to knowledge base: {e}")
            return False
    
    def get_knowledge_base_stats(self) -> dict:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with stats
        """
        content = self.load_knowledge_base()
        
        stats = {
            "total_characters": len(content),
            "total_words": len(content.split()),
            "total_lines": len(content.split('\n')),
            "exists": os.path.exists(self.kb_path),
            "file_size_kb": os.path.getsize(self.kb_path) / 1024 if os.path.exists(self.kb_path) else 0
        }
        
        return stats
    
    def load_chunks(self) -> Optional[List[str]]:
        """
        Load pre-chunked documents.
        
        Returns:
            List of chunks or None if not found
        """
        try:
            if not os.path.exists(self.chunks_path):
                logger.warning(f"Chunks file not found at {self.chunks_path}")
                return None
            
            with open(self.chunks_path, 'rb') as f:
                chunks = pickle.load(f)
            
            logger.info(f"✅ Loaded {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error loading chunks: {e}")
            return None
    
    def save_chunks(self, chunks: List[str]) -> bool:
        """
        Save chunks to file.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(chunks, f)
            
            logger.info(f"✅ Saved {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving chunks: {e}")
            return False
    
    def display_stats(self):
        """Display knowledge base statistics."""
        stats = self.get_knowledge_base_stats()
        
        print("\n" + "="*50)
        print("📊 Knowledge Base Statistics")
        print("="*50)
        print(f"Exists: {stats['exists']}")
        print(f"Size: {stats['file_size_kb']:.2f} KB")
        print(f"Characters: {stats['total_characters']:,}")
        print(f"Words: {stats['total_words']:,}")
        print(f"Lines: {stats['total_lines']:,}")
        print("="*50 + "\n")


class ContentOrganizer:
    """Organize knowledge base content into categories."""
    
    def __init__(self, kb_manager: KnowledgeBaseManager):
        self.kb_manager = kb_manager
    
    def organize_by_sections(self, separator: str = "---") -> dict:
        """
        Organize content into sections.
        
        Args:
            separator: Section separator string
            
        Returns:
            Dictionary of sections
        """
        content = self.kb_manager.load_knowledge_base()
        
        if not content:
            return {}
        
        sections = {}
        current_section = "default"
        current_content = []
        
        for line in content.split('\n'):
            if line.strip().startswith(separator):
                if current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.replace(separator, "").strip() or f"section_{len(sections)}"
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def display_sections(self):
        """Display organized sections."""
        sections = self.organize_by_sections()
        
        print("\n" + "="*50)
        print("📚 Knowledge Base Sections")
        print("="*50)
        for name, content in sections.items():
            words = len(content.split())
            print(f"- {name}: {words} words")
        print("="*50 + "\n")


def demonstrate_knowledge_base_management():
    """Demonstrate knowledge base management."""
    
    # Create manager
    manager = KnowledgeBaseManager()
    
    # Display stats
    manager.display_stats()
    
    # Organize sections
    organizer = ContentOrganizer(manager)
    organizer.display_sections()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_knowledge_base_management()
