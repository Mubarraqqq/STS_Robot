"""
Prompt Management System
Handles system prompts and dynamic prompt generation based on context
"""

from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PromptTemplate:
    """Base class for prompt templates."""
    
    def __init__(self, name: str, template: str):
        self.name = name
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable in template: {e}")
            return self.template


class SystemPrompt:
    """System prompt for the assistant."""
    
    SYSTEM_CONTENT = """You are Bruce, a helpful and knowledgeable voice assistant. 

**Core Personality:**
- Friendly, conversational, and approachable
- Clear and concise in your responses
- Respectful and patient with users
- Professional yet personable

**Response Guidelines:**
- Keep responses brief and suitable for voice output (30-60 seconds of speech)
- Use simple, everyday language that's easy to understand
- Avoid technical jargon unless necessary
- Be informative but not overwhelming
- No emojis or special symbols in output
- Structure longer answers with pauses between sentences

**Behavior Rules:**
- Always acknowledge what you understand from the user's question
- If unsure about something, admit it rather than guess
- Offer to help with related questions
- Stay on topic and redirect politely if needed
- Remember conversation context from recent exchanges"""
    
    @staticmethod
    def get_content() -> str:
        """Get the system prompt content."""
        return SystemPrompt.SYSTEM_CONTENT


class PromptGenerator:
    """Generates prompts based on RAG confidence levels."""
    
    # Confidence levels and their prompt templates
    TEMPLATES = {
        "HIGH": PromptTemplate(
            "high_confidence",
            """Based on the following information, answer the user's question accurately and completely.

**Context:**
{context}

**User Question:** {question}

**Instructions:**
1. Answer strictly based on the provided context
2. Be thorough but concise
3. If context contains relevant details, include them
4. If something in the context directly answers the question, prioritize that"""
        ),
        "MEDIUM": PromptTemplate(
            "medium_confidence",
            """Use the following information to help answer the user's question. The match may not be perfect, so use your judgment.

**Available Context:**
{context}

**User Question:** {question}

**Instructions:**
1. Use the context to guide your answer
2. Acknowledge any uncertainty if the context isn't a perfect match
3. Combine context with your general knowledge if helpful
4. Be clear about what comes from the provided context"""
        ),
        "LOW": PromptTemplate(
            "low_confidence",
            """Answer the user's question using your general knowledge and reasoning abilities.

**User Question:** {question}

**Instructions:**
1. Provide a helpful, accurate answer
2. Feel free to use your general knowledge
3. Be conversational and natural
4. Ask clarifying questions if needed"""
        ),
    }
    
    @staticmethod
    def generate(
        confidence_level: str,
        question: str,
        context: Optional[str] = None
    ) -> str:
        """
        Generate a prompt based on confidence level.
        
        Args:
            confidence_level: "HIGH", "MEDIUM", or "LOW"
            question: User's question
            context: Retrieved context (optional, required for HIGH/MEDIUM)
            
        Returns:
            Formatted prompt string
        """
        if confidence_level not in PromptGenerator.TEMPLATES:
            logger.warning(f"Unknown confidence level: {confidence_level}, defaulting to LOW")
            confidence_level = "LOW"
        
        template = PromptGenerator.TEMPLATES[confidence_level]
        
        if confidence_level in ["HIGH", "MEDIUM"]:
            if not context:
                logger.warning(f"No context provided for {confidence_level} confidence level")
                context = "(No additional context available)"
            
            return template.format(context=context, question=question)
        else:
            return template.format(question=question)


class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 4):
        """
        Initialize conversation manager.
        
        Args:
            max_history: Maximum number of messages to keep in history
        """
        self.history: List[dict] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        """
        Add a message to history.
        
        Args:
            role: "user" or "assistant"
            content: Message content
        """
        self.history.append({"role": role, "content": content})
        
        # Keep only recent messages
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.debug(f"Added {role} message, history now has {len(self.history)} messages")
    
    def get_history(self) -> List[dict]:
        """Get conversation history."""
        return self.history.copy()
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
        logger.info("Conversation history cleared")
    
    def format_for_api(self) -> List[dict]:
        """
        Format history for API consumption.
        
        Returns:
            List of message dictionaries suitable for API calls
        """
        return self.get_history()
    
    def get_context_summary(self) -> str:
        """
        Get a summary of recent conversation context.
        
        Returns:
            String summarizing recent exchanges
        """
        if not self.history:
            return "No recent conversation context."
        
        summary = "Recent conversation:\n"
        for msg in self.history[-4:]:  # Last 2 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long messages
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            summary += f"- {role}: {content}\n"
        
        return summary


class RAGPromptBuilder:
    """Builds complete prompts for RAG-based responses."""
    
    @staticmethod
    def build_rag_prompt(
        question: str,
        retrieved_chunks: List[Tuple[str, float]],
        conversation_history: Optional[List[dict]] = None,
        similarity_threshold: float = 0.80
    ) -> Tuple[str, str]:
        """
        Build a complete prompt with RAG context.
        
        Args:
            question: User's question
            retrieved_chunks: List of (chunk, similarity) tuples
            conversation_history: Previous conversation messages
            similarity_threshold: Minimum similarity for "HIGH" confidence
            
        Returns:
            Tuple of (prompt, confidence_level)
        """
        
        if not retrieved_chunks:
            confidence = "LOW"
            prompt = PromptGenerator.generate(confidence, question)
            return prompt, confidence
        
        best_chunk, best_similarity = retrieved_chunks[0]
        
        # Determine confidence level
        if best_similarity >= 0.85:
            confidence = "HIGH"
            # Use top 2 chunks for high confidence
            context = "\n\n---\n\n".join([chunk for chunk, _ in retrieved_chunks[:2]])
        elif best_similarity >= 0.75:
            confidence = "MEDIUM"
            # Use top chunk for medium confidence
            context = best_chunk
        else:
            confidence = "LOW"
            context = None
        
        prompt = PromptGenerator.generate(confidence, question, context)
        
        # Add conversation context if available
        if conversation_history:
            prompt = RAGPromptBuilder._add_history_context(
                prompt,
                conversation_history
            )
        
        return prompt, confidence
    
    @staticmethod
    def _add_history_context(prompt: str, history: List[dict]) -> str:
        """
        Add conversation history context to prompt.
        
        Args:
            prompt: Base prompt
            history: Conversation history
            
        Returns:
            Enhanced prompt with history context
        """
        if len(history) >= 2:
            history_str = "\n".join([
                f"- {msg['role'].capitalize()}: {msg['content'][:100]}"
                for msg in history[-2:]
            ])
            prompt = f"""**Recent Context:**
{history_str}

{prompt}"""
        
        return prompt
