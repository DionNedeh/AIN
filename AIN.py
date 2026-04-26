import gc
import json
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from llama_cpp import Llama


class AILocalNotepad:
    MODEL_PATH = r"C:\Users\.lmstudio\models\lmstudio-community\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-Q4_K_M.gguf"   # Replace with actual model on your system
    DEFAULT_SYSTEM_PROMPT = "You are a helpful and concise AI assistant embedded in a notepad application."
    MESSAGE_TOKEN_OVERHEAD = 6
    
    def __init__(self, root):
        self.root = root
        self.root.title("AIN")
        self.root.geometry("1200x800")
        
        # State variables
        self.current_file = None
        self.is_dark_theme = True
        self.llm = None
        self.chat_history = [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]
        
        self.history_dir = Path.home() / "AIN_chats"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        self.loading_model = False
        self.generating_response = False
        self.stop_event = threading.Event()
        
        self._build_menu()
        self._build_ui()
        self._apply_theme()
        
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New File", command=self.new_file)
        file_menu.add_command(label="Open File", command=self.open_file)
        file_menu.add_command(label="Save File", command=self.save_file)
        file_menu.add_command(label="Save File As", command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Toggle Theme", command=self.toggle_theme)
        menubar.add_cascade(label="View", menu=view_menu)
        
        self.root.config(menu=menubar)

    def _build_ui(self):
        # PanedWindow to separate Notepad and Chat
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left side: Notepad
        self.notepad_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.notepad_frame, weight=3)
        
        self.text_editor = scrolledtext.ScrolledText(self.notepad_frame, wrap=tk.WORD, font=("Consolas", 12))
        self.text_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right side: Chat interface
        self.chat_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.chat_frame, weight=1)
        
        # System Prompt
        sys_prompt_frame = ttk.LabelFrame(self.chat_frame, text="System Prompt")
        sys_prompt_frame.pack(fill=tk.X, padx=5, pady=5)
        self.sys_prompt_text = scrolledtext.ScrolledText(sys_prompt_frame, height=4, wrap=tk.WORD, font=("Segoe UI", 9))
        self.sys_prompt_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.sys_prompt_text.insert(tk.END, self.DEFAULT_SYSTEM_PROMPT)
        
        # Chat Controls
        controls_frame = ttk.Frame(self.chat_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Row 1: Engine controls
        engine_frame = ttk.Frame(controls_frame)
        engine_frame.pack(fill=tk.X, pady=2)
        self.connect_btn = ttk.Button(engine_frame, text="Connect AI", command=self.load_model_thread)
        self.connect_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.disconnect_btn = ttk.Button(engine_frame, text="Disconnect", command=self.deload_model, state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Row 2: Chat controls
        chat_btns_frame = ttk.Frame(controls_frame)
        chat_btns_frame.pack(fill=tk.X, pady=2)
        self.new_chat_btn = ttk.Button(chat_btns_frame, text="New Chat", command=self.new_chat)
        self.new_chat_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.save_chat_btn = ttk.Button(chat_btns_frame, text="Save Chat", command=self.save_chat)
        self.save_chat_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.load_chat_btn = ttk.Button(chat_btns_frame, text="Load Chat", command=self.load_chat)
        self.load_chat_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # Status and Tokens
        self.status_var = tk.StringVar(value="Status: Offline")
        self.status_label = ttk.Label(self.chat_frame, textvariable=self.status_var, font=("Segoe UI", 9, "bold"))
        self.status_label.pack(fill=tk.X, padx=5)

        self.token_stats_var = tk.StringVar(value="Usage (Est.): prompt 0 | completion 0 | total 0 | ctx 0.0%")
        self.token_label = ttk.Label(self.chat_frame, textvariable=self.token_stats_var, font=("Segoe UI", 8))
        self.token_label.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # Chat Display
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, state="disabled", wrap=tk.WORD, font=("Segoe UI", 10))
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Chat Input
        input_frame = ttk.Frame(self.chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.user_input = ttk.Entry(input_frame, font=("Segoe UI", 11))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", lambda event: self.send_message())
        
        self.send_btn = ttk.Button(input_frame, text="Send", command=self.send_message, state=tk.DISABLED)
        self.send_btn.pack(side=tk.LEFT, padx=(0, 2))
        
        self.stop_btn = ttk.Button(input_frame, text="Stop", command=self.stop_generation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        self.update_chat_display("System: Ready. Click 'Connect AI' to load the model.\n\n")

    def _apply_theme(self):
        if self.is_dark_theme:
            bg_color = "#1e1e1e"
            fg_color = "#d4d4d4"
            insert_color = "white"
        else:
            bg_color = "#ffffff"
            fg_color = "#000000"
            insert_color = "black"
            
        self.text_editor.config(bg=bg_color, fg=fg_color, insertbackground=insert_color)
        self.chat_display.config(bg=bg_color, fg=fg_color)
        self.sys_prompt_text.config(bg=bg_color, fg=fg_color, insertbackground=insert_color)

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        self._apply_theme()

    # --- File Operations ---
    def new_file(self):
        self.text_editor.delete("1.0", tk.END)
        self.current_file = None
        self.root.title("AI Notepad - Untitled")

    def open_file(self):
        filepath = filedialog.askopenfilename(defaultextension=".txt",
                                              filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if filepath:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text_editor.delete("1.0", tk.END)
                self.text_editor.insert(tk.END, content)
                self.current_file = filepath
                self.root.title(f"AI Notepad - {Path(filepath).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not read file: {e}")

    def save_file(self):
        if self.current_file:
            try:
                content = self.text_editor.get("1.0", tk.END)
                with open(self.current_file, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")
        else:
            self.save_as_file()

    def save_as_file(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if filepath:
            try:
                content = self.text_editor.get("1.0", tk.END)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                self.current_file = filepath
                self.root.title(f"AI Notepad - {Path(filepath).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")

    # --- Chat History Management ---
    def new_chat(self):
        if self.generating_response:
            messagebox.showinfo("Busy", "Please stop or wait for the current response to complete.")
            return

        sys_prompt = self.sys_prompt_text.get("1.0", tk.END).strip()
        self.chat_history = [{"role": "system", "content": sys_prompt}]

        self.chat_display.config(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state="disabled")

        self._update_usage_stats(0, 0)
        self.update_chat_display("System: Chat cleared.\n\n")

    def save_chat(self):
        filepath = filedialog.asksaveasfilename(
            initialdir=str(self.history_dir),
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save Chat History",
        )
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(self.chat_history, f, indent=4, ensure_ascii=False)
                self.update_chat_display(f"System: Chat saved to {Path(filepath).name}\n\n")
            except OSError as e:
                messagebox.showerror("Save Error", f"Could not save chat: {e}")

    def load_chat(self):
        if self.generating_response:
            messagebox.showinfo("Busy", "Please stop or wait for the current response to complete.")
            return

        filepath = filedialog.askopenfilename(
            initialdir=str(self.history_dir),
            filetypes=[("JSON Files", "*.json")],
            title="Load Chat History",
        )
        if filepath:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded_history = json.load(f)

                if not isinstance(loaded_history, list):
                    raise ValueError("Invalid chat history format.")
            except (OSError, json.JSONDecodeError, ValueError) as e:
                messagebox.showerror("Load Error", f"Could not load chat: {e}")
                return

            self.chat_history = loaded_history

            self.chat_display.config(state="normal")
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state="disabled")

            if self.chat_history and self.chat_history[0].get("role") == "system":
                self.sys_prompt_text.delete("1.0", tk.END)
                self.sys_prompt_text.insert(tk.END, self.chat_history[0].get("content", ""))

            for msg in self.chat_history:
                role = "You" if msg["role"] == "user" else "AI" if msg["role"] == "assistant" else "System"
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "\n".join([part.get("text", "") for part in content if part.get("type") == "text"])
                if role != "System":
                    self.update_chat_display(f"{role}: {content}\n\n")

            self.update_chat_display("System: Chat history loaded successfully.\n\n")

    # --- Token Estimation ---
    def _estimate_text_tokens(self, text):
        if not text:
            return 0
        if self.llm is None:
            return max(1, len(text) // 4)
        try:
            return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
        except Exception:
            return max(1, len(text) // 4)

    def _estimate_content_tokens(self, content):
        if isinstance(content, str):
            return self._estimate_text_tokens(content)
        if isinstance(content, list):
            total = 0
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += self._estimate_text_tokens(part.get("text", ""))
            return total
        return self._estimate_text_tokens(str(content))

    def _estimate_messages_tokens(self, messages):
        total = 0
        for msg in messages:
            total += self.MESSAGE_TOKEN_OVERHEAD
            total += self._estimate_content_tokens(msg.get("content", ""))
        return total + 3

    def _update_usage_stats(self, prompt_tokens, completion_tokens):
        total_tokens = max(0, prompt_tokens) + max(0, completion_tokens)

        context_size = 131072
        if self.llm is not None:
            try:
                context_size = int(self.llm.n_ctx())
            except Exception:
                pass

        context_pct = (total_tokens / context_size * 100.0) if context_size > 0 else 0.0
        self.token_stats_var.set(
            f"Usage (Est.): prompt {prompt_tokens} | completion {completion_tokens} | total {total_tokens} | ctx {context_pct:.1f}%"
        )

    # --- AI Integration ---
    def load_model_thread(self):
        if self.loading_model or self.llm is not None:
            return
            
        if not os.path.exists(self.MODEL_PATH):
            messagebox.showerror("Model Error", f"Model not found at:\n{self.MODEL_PATH}")
            return

        self.loading_model = True
        self.status_var.set("Status: Loading model...")
        self.connect_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        
        self.update_chat_display("System: Allocating memory... please wait.\n\n")
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            llm = Llama(
                model_path=self.MODEL_PATH,
                n_ctx=131072,
                n_gpu_layers=-1,
                verbose=False
            )
            self.llm = llm
            self.root.after(0, self._on_model_loaded)
        except Exception as e:
            self.llm = None
            self.root.after(0, self._on_model_load_failure, str(e))

    def _on_model_loaded(self):
        self.loading_model = False
        self.status_var.set("Status: Online")
        self.connect_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.NORMAL)
        self.update_chat_display("System: Engine Online. Ready to chat.\n\n")

    def _on_model_load_failure(self, error_message):
        self.loading_model = False
        self.status_var.set("Status: Load Failure")
        self.connect_btn.config(state=tk.NORMAL)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.update_chat_display(f"Critical Error: {error_message}\n\n")

    def deload_model(self):
        if self.loading_model or self.generating_response:
            messagebox.showinfo("Busy", "Please wait for the current operation to finish.")
            return

        if self.llm:
            del self.llm
            self.llm = None
            gc.collect()
            
            self.status_var.set("Status: Offline")
            self.connect_btn.config(state=tk.NORMAL)
            self.disconnect_btn.config(state=tk.DISABLED)
            self.send_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            
            self._update_usage_stats(0, 0)
            self.update_chat_display("System: Model deloaded and memory purged.\n\n")

    def stop_generation(self):
        if not self.generating_response:
            return
        self.stop_event.set()
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Status: Stopping...")

    def send_message(self):
        if not self.llm or self.loading_model or self.generating_response:
            return

        message = self.user_input.get().strip()
        if not message:
            return

        self.user_input.delete(0, tk.END)
        self.generating_response = True
        self.stop_event.clear()
        
        # Update system prompt dynamically if it was changed
        sys_prompt = self.sys_prompt_text.get("1.0", tk.END).strip()
        if self.chat_history and self.chat_history[0]["role"] == "system":
            self.chat_history[0]["content"] = sys_prompt
        elif not self.chat_history:
            self.chat_history.append({"role": "system", "content": sys_prompt})
        
        self.send_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        self.update_chat_display(f"You: {message}\n\n")
        self.chat_history.append({"role": "user", "content": message})
        
        prompt_token_estimate = self._estimate_messages_tokens(self.chat_history)
        
        self.update_chat_display("AI: ")
        threading.Thread(target=self.generate_response, args=(prompt_token_estimate,), daemon=True).start()

    def generate_response(self, prompt_token_estimate):
        stream = None
        full_reply = ""
        stopped = False

        try:
            stream = self.llm.create_chat_completion(
                messages=self.chat_history,
                max_tokens=1024,
                stream=True
            )

            for chunk in stream:
                if self.stop_event.is_set():
                    stopped = True
                    break

                text_chunk = self._extract_stream_text(chunk)
                if text_chunk:
                    full_reply += text_chunk
                    self.root.after(0, self.update_chat_display, text_chunk)

            if stopped:
                if full_reply:
                    self.root.after(0, self.update_chat_display, "\n[Generation stopped]\n\n")
                else:
                    full_reply = "[Generation stopped by user.]"
                    self.root.after(0, self.update_chat_display, full_reply + "\n\n")
            else:
                self.root.after(0, self.update_chat_display, "\n\n")

            if full_reply:
                self.chat_history.append({"role": "assistant", "content": full_reply})
                self.root.after(0, self._add_response_actions, full_reply)

            completion_tokens = self._estimate_text_tokens(full_reply)
            self.root.after(0, self._update_usage_stats, prompt_token_estimate, completion_tokens)

        except Exception as e:
            self.root.after(0, self.update_chat_display, f"\n[Error: {str(e)}]\n\n")
        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:
                    pass
            self.root.after(0, self._unlock_after_response)

    @staticmethod
    def _extract_stream_text(chunk):
        if not isinstance(chunk, dict):
            return ""
        choices = chunk.get("choices")
        if not choices or not isinstance(choices, list):
            return ""
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""
        delta = first_choice.get("delta")
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content")
        return content if isinstance(content, str) else ""

    def _add_response_actions(self, text):
        # Create a small frame to hold the action buttons inside the text widget
        action_frame = ttk.Frame(self.chat_display)
        
        copy_btn = ttk.Button(action_frame, text="Copy", command=lambda t=text: self._copy_to_clipboard(t))
        copy_btn.pack(side=tk.LEFT, padx=2)
        
        insert_btn = ttk.Button(action_frame, text="Insert", command=lambda t=text: self._insert_to_notepad(t))
        insert_btn.pack(side=tk.LEFT, padx=2)
        
        self.chat_display.config(state="normal")
        self.chat_display.window_create(tk.END, window=action_frame)
        self.chat_display.insert(tk.END, "\n\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def _copy_to_clipboard(self, text):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()

    def _insert_to_notepad(self, text):
        self.text_editor.insert(tk.INSERT, text)
        self.text_editor.see(tk.INSERT)

    def _unlock_after_response(self):
        self.generating_response = False
        self.stop_event.clear()
        self.stop_btn.config(state=tk.DISABLED)
        
        if self.llm and not self.loading_model:
            self.send_btn.config(state=tk.NORMAL)
            self.status_var.set("Status: Online")

    def update_chat_display(self, text):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def _on_close(self):
        if self.generating_response:
            self.stop_event.set()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    
    # Try applying azure theme if available, otherwise fallback to default
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "dark")
    except tk.TclError:
        pass
        
    app = AILocalNotepad(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()
