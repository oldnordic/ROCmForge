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
        "gemma" | "gemma2" | "gemma3" => ChatTemplate::Gemma,
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
}
