import json
import os
from typing import List, Dict, Optional


class ConversationHistory:
    def __init__(self, persist_path: Optional[str] = None):
        self.history: List[Dict[str, str]] = []
        self.persist_path = persist_path
        if self.persist_path:
            self._load()

    def add_turn(self, user_input, bot_response):
        self.history.append({'user_input': user_input, 'bot_response': bot_response})
        self._save()

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []
        self._save()

    def get_last_n_turns(self, n):
        return self.history[-n:] if n <= len(self.history) else self.history

    def get_formatted_history(self):
        formatted_history = ""
        for turn in self.history:
            formatted_history += f"User: {turn['user_input']}\nBot: {turn['bot_response']}\n"
        return formatted_history.strip()

    def _save(self):
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(self.persist_path):
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
        except Exception:
            self.history = []
