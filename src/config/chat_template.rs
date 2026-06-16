/// Prompt wrapping format for instruction-tuned models.
///
/// Each variant corresponds to one architecture family's instruct format.
/// Selection uses both the GGUF architecture string AND the tokenizer type,
/// because LLaMA2 and LLaMA3 share `architecture = "llama"` but use
/// different templates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// No wrapping - raw completion mode.
    None,
    /// Qwen2/3 ChatML: `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`
    ChatML,
    /// LLaMA3: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>`
    LLaMA3,
    /// LLaMA2 / Mistral v0.1: `[INST] ... [/INST]`
    LLaMA2,
    /// Phi3: `<|user|>\n...<|end|>\n<|assistant|">\n`
    Phi3,
    /// Gemma: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
    Gemma,
}

impl ChatTemplate {
    /// Wrap `user_text` in the appropriate prompt format.
    /// Returns the text unchanged when `self == None`.
    pub fn apply(&self, user_text: &str) -> String {
        match self {
            ChatTemplate::None => user_text.to_string(),

            ChatTemplate::ChatML => format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                user_text
            ),

            ChatTemplate::LLaMA3 => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                user_text
            ),

            ChatTemplate::LLaMA2 => format!("[INST] {} [/INST]", user_text),

            ChatTemplate::Phi3 => format!(
                "<|user|>\n{}<|end|>\n<|assistant|\">\n",
                user_text
            ),

            ChatTemplate::Gemma => format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                user_text
            ),
        }
    }

    pub fn apply_messages(&self, messages: &[(String, String)]) -> String {
        match self {
            ChatTemplate::None => messages
                .iter()
                .map(|(role, text)| format!("{}: {}", role, text))
                .collect::<Vec<_>>()
                .join("\n"),

            ChatTemplate::ChatML => {
                messages
                    .iter()
                    .map(|(role, text)| format!("<|im_start|>{}\n{}<|im_end|>", role, text))
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n<|im_start|>assistant\n"
            }

            ChatTemplate::LLaMA3 => {
                let parts: Vec<String> = messages
                    .iter()
                    .map(|(role, text)| {
                        let r = match role.as_str() {
                            "system" => "system",
                            "user" => "user",
                            _ => "assistant",
                        };
                        format!(
                            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                            r, text
                        )
                    })
                    .collect();
                let mut out = String::new();
                out.push_str("<|begin_of_text|>");
                for p in &parts {
                    out.push_str(p);
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                out
            }

            ChatTemplate::LLaMA2 => {
                let mut out = String::new();
                let mut first_user = true;
                for (role, text) in messages {
                    match role.as_str() {
                        "system" => {
                            out.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", text))
                        }
                        "user" if first_user => {
                            if out.is_empty() {
                                out.push_str(&format!("[INST] {} [/INST]", text));
                            } else {
                                out.push_str(&format!("{} [/INST]", text));
                            }
                            first_user = false;
                        }
                        "user" => out.push_str(&format!(" [INST] {} [/INST]", text)),
                        "assistant" => out.push_str(&format!(" {} ", text)),
                        _ => out.push_str(&format!(" {}: {} ", role, text)),
                    }
                }
                if !out.ends_with("[/INST]") {
                    out.push_str(" [/INST]");
                }
                // Strip trailing INST and add assistant
                if let Some(idx) = out.rfind("[/INST]") {
                    out.truncate(idx + 7);
                }
                out.push(' ');
                out
            }

            ChatTemplate::Phi3 => {
                messages
                    .iter()
                    .map(|(role, text)| format!("<|{}|>\n{}<|end|>", role, text))
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n<|assistant|>\n"
            }

            ChatTemplate::Gemma => {
                messages
                    .iter()
                    .map(|(role, text)| {
                        let r = if role == "assistant" { "model" } else { role };
                        format!("<start_of_turn>{}\n{}<end_of_turn>", r, text)
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n<start_of_turn>model\n"
            }
        }
    }

    /// Human-readable name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            ChatTemplate::None => "none (raw completion)",
            ChatTemplate::ChatML => "ChatML (Qwen2/3)",
            ChatTemplate::LLaMA3 => "LLaMA3",
            ChatTemplate::LLaMA2 => "LLaMA2/Mistral",
            ChatTemplate::Phi3 => "Phi3",
            ChatTemplate::Gemma => "Gemma",
        }
    }
}

/// Detect the appropriate chat template from architecture + tokenizer type.
///
/// `tokenizer_model` is the value of `tokenizer.ggml.model` from the GGUF
/// KV section (e.g. `"gpt2"`, `"llama"`, `"spm"`).
///
/// The distinction between LLaMA2 and LLaMA3 requires the tokenizer type:
/// - LLaMA3 uses BPE (`"gpt2"`) with 128K vocab
/// - LLaMA2 uses SentencePiece (`"llama"` / `"spm"`)
pub fn detect_chat_template(architecture: &str, tokenizer_model: Option<&str>) -> ChatTemplate {
    match architecture {
        "qwen2" | "qwen3" | "qwen2moe" | "qwen3moe" | "qwen" => ChatTemplate::ChatML,

        "llama" | "mistral" | "yi" | "baichuan" | "internlm2" | "deepseek" => {
            // Distinguish LLaMA3 (BPE) from LLaMA2/Mistral (SPM)
            match tokenizer_model {
                Some("gpt2") | Some("bpe") => ChatTemplate::LLaMA3,
                _ => ChatTemplate::LLaMA2,
            }
        }

        "phi3" => ChatTemplate::Phi3,
        "gemma" | "gemma2" | "gemma3" | "gemma4" => ChatTemplate::Gemma,
        "mixtral" => ChatTemplate::LLaMA2,

        // Unknown architecture: no template, raw completion
        _ => ChatTemplate::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_qwen2_is_chatml() {
        assert_eq!(
            detect_chat_template("qwen2", Some("gpt2")),
            ChatTemplate::ChatML
        );
        assert_eq!(detect_chat_template("qwen3", None), ChatTemplate::ChatML);
    }

    #[test]
    fn template_llama3_detected_by_bpe() {
        // LLaMA3 has BPE tokenizer
        assert_eq!(
            detect_chat_template("llama", Some("gpt2")),
            ChatTemplate::LLaMA3
        );
    }

    #[test]
    fn template_llama2_detected_by_spm() {
        // LLaMA2 / Mistral use SentencePiece
        assert_eq!(
            detect_chat_template("llama", Some("llama")),
            ChatTemplate::LLaMA2
        );
        assert_eq!(
            detect_chat_template("mistral", Some("llama")),
            ChatTemplate::LLaMA2
        );
        assert_eq!(detect_chat_template("mixtral", None), ChatTemplate::LLaMA2);
    }

    #[test]
    fn template_phi3() {
        assert_eq!(detect_chat_template("phi3", None), ChatTemplate::Phi3);
    }

    #[test]
    fn template_gemma() {
        assert_eq!(detect_chat_template("gemma2", None), ChatTemplate::Gemma);
    }

    #[test]
    fn template_unknown_arch_is_none() {
        assert_eq!(
            detect_chat_template("future_arch", None),
            ChatTemplate::None
        );
    }

    #[test]
    fn chatml_apply_wraps_correctly() {
        let t = ChatTemplate::ChatML;
        let out = t.apply("Hello");
        assert!(out.starts_with("<|im_start|>user\n"));
        assert!(out.contains("Hello"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama2_apply() {
        let out = ChatTemplate::LLaMA2.apply("Hi");
        assert_eq!(out, "[INST] Hi [/INST]");
    }

    #[test]
    fn none_apply_passthrough() {
        let text = "raw prompt";
        assert_eq!(ChatTemplate::None.apply(text), text);
    }

    #[test]
    fn chatml_multi_turn() {
        let msgs = vec![
            ("system".to_string(), "You are helpful.".to_string()),
            ("user".to_string(), "Hello".to_string()),
        ];
        let out = ChatTemplate::ChatML.apply_messages(&msgs);
        assert!(out.contains("<|im_start|>system"));
        assert!(out.contains("<|im_start|>user"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn llama3_roundtrip_multi_turn() {
        let msgs = vec![
            ("system".to_string(), "SYS".to_string()),
            ("user".to_string(), "Hi".to_string()),
        ];
        let out = ChatTemplate::LLaMA3.apply_messages(&msgs);
        assert!(out.starts_with("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>system"));
        assert!(out.contains("<|start_header_id|>user"));
        assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn llama2_roundtrip_multi_turn() {
        let msgs = vec![
            ("user".to_string(), "First".to_string()),
            ("assistant".to_string(), "Ok".to_string()),
            ("user".to_string(), "Second".to_string()),
        ];
        let out = ChatTemplate::LLaMA2.apply_messages(&msgs);
        // LLaMA2 flattens to a single [INST] block.
        assert!(out.starts_with("[INST]"));
        assert!(out.contains("Second"));
    }

    #[test]
    fn phi3_multi_turn() {
        let msgs = vec![("user".to_string(), "Q".to_string())];
        let out = ChatTemplate::Phi3.apply_messages(&msgs);
        assert!(out.contains("<|user|>"));
        assert!(out.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn gemma_multi_turn_remaps_assistant() {
        let msgs = vec![
            ("user".to_string(), "Q".to_string()),
            ("assistant".to_string(), "A".to_string()),
        ];
        let out = ChatTemplate::Gemma.apply_messages(&msgs);
        assert!(out.contains("<start_of_turn>user"));
        assert!(out.contains("<start_of_turn>model"));
        assert!(!out.contains("<start_of_turn>assistant"));
    }
}
