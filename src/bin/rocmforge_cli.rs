use clap::{Parser, Subcommand};
use futures::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
use rocmforge::engine::{EngineConfig, InferenceEngine};
use rocmforge::http::server::run_server;
use rocmforge::models::discover_models;
use rocmforge::tokenizer::{embedded_tokenizer_from_gguf, infer_tokenizer_path, TokenizerAdapter};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{self, AsyncWriteExt};
use tokio::time::sleep;

#[derive(Parser, Debug)]
#[command(name = "rocmforge-cli", version)]
#[command(about = "Interact with a running ROCmForge inference server", long_about = None)]
struct Cli {
    /// Base URL of the ROCmForge HTTP server
    #[arg(long, default_value = "http://127.0.0.1:8080")]
    host: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the built-in HTTP server (similar to `ollama serve`)
    Serve {
        /// Address to bind the HTTP server to
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Path to the GGUF model to load
        #[arg(long)]
        gguf: Option<String>,
        /// Path to tokenizer JSON (defaults to ROCMFORGE_TOKENIZER env or fallback)
        #[arg(long)]
        tokenizer: Option<String>,
    },
    /// Generate text once and print the final response
    Generate {
        /// Prompt text to feed the model
        #[arg(short, long)]
        prompt: String,
        /// Maximum number of new tokens to generate
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Sampling temperature
        #[arg(long)]
        temperature: Option<f32>,
        /// Top-k sampling
        #[arg(long)]
        top_k: Option<usize>,
        /// Top-p sampling
        #[arg(long)]
        top_p: Option<f32>,
        /// Stream tokens as they are generated
        #[arg(long)]
        stream: bool,
        /// Run inference locally using the specified GGUF model instead of HTTP
        #[arg(long)]
        gguf: Option<String>,
        /// Path to tokenizer JSON to use in local mode
        #[arg(long)]
        tokenizer: Option<String>,
    },
    /// Query request status by id
    Status {
        /// Request identifier returned by a previous generate command
        #[arg(long)]
        request_id: u32,
    },
    /// Cancel a running request on the HTTP server
    Cancel {
        /// Request identifier to cancel
        #[arg(long)]
        request_id: u32,
    },
    /// List available GGUF models
    Models {
        /// Directory containing GGUF files (defaults to ROCMFORGE_MODELS or ./models)
        #[arg(long)]
        dir: Option<String>,
    },
}

#[derive(Debug, Serialize, Clone)]
struct GenerateRequest {
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    request_id: u32,
    text: String,
    tokens: Vec<u32>,
    finished: bool,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TokenStream {
    request_id: u32,
    token: u32,
    text: String,
    finished: bool,
    finish_reason: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Generate {
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            stream,
            gguf,
            tokenizer,
        } => {
            let request = GenerateRequest {
                prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                stream: Some(stream),
            };
            if let Some(path) = gguf {
                let tokenizer_path = tokenizer.clone().or_else(|| infer_tokenizer_path(&path));
                let embedded_tokenizer = if tokenizer_path.is_none() {
                    embedded_tokenizer_from_gguf(&path)
                } else {
                    None
                };
                if tokenizer_path.is_none() {
                    if let Some(info) = &embedded_tokenizer {
                        eprintln!(
                            "INFO: using tokenizer embedded in GGUF (cached: {}, refreshed: {})",
                            info.cached, info.refreshed
                        );
                    } else {
                        eprintln!("WARN: tokenizer not provided and no tokenizer.json found; using fallback hashing tokenizer");
                    }
                }
                let tokenizer = TokenizerAdapter::from_spec(
                    tokenizer_path.as_deref(),
                    embedded_tokenizer.as_ref().map(|t| t.json.as_str()),
                );
                if stream {
                    run_local_stream(&path, &tokenizer, &request).await?;
                } else {
                    run_local_generate(&path, &tokenizer, &request).await?;
                }
            } else if stream {
                run_http_stream(&cli.host, request).await?;
            } else {
                run_http_generate(&cli.host, request).await?;
            }
        }
        Commands::Status { request_id } => {
            fetch_status(&cli.host, request_id).await?;
        }
        Commands::Cancel { request_id } => {
            cancel_http_request(&cli.host, request_id).await?;
        }
        Commands::Serve {
            addr,
            gguf,
            tokenizer,
        } => {
            let tokenizer_path = tokenizer
                .clone()
                .or_else(|| gguf.as_deref().and_then(infer_tokenizer_path));
            if tokenizer_path.is_none() {
                eprintln!(
                    "WARN: tokenizer not provided and no tokenizer file inferred; HTTP server will use fallback tokenizer"
                );
            }
            run_server(&addr, gguf.as_deref(), tokenizer_path.as_deref()).await?;
        }
        Commands::Models { dir } => {
            list_models(dir.as_deref()).await?;
        }
    }
    Ok(())
}

async fn run_http_generate(host: &str, body: GenerateRequest) -> anyhow::Result<()> {
    let client = Client::new();
    let url = format!("{}/generate", host.trim_end_matches('/'));
    let resp = client.post(url).json(&body).send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await
            .unwrap_or_else(|e| format!("<failed to read error body: {}>", e));
        anyhow::bail!("Server returned error {}: {}", status, text);
    }
    let response: GenerateResponse = resp.json().await?;
    println!("request_id: {}", response.request_id);
    println!("finish_reason: {:?}", response.finish_reason);
    println!("text:\n{}", response.text);
    Ok(())
}

async fn run_http_stream(host: &str, body: GenerateRequest) -> anyhow::Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let url = format!("{}/generate/stream", host.trim_end_matches('/'));
    let mut es = EventSource::new(client.post(url).json(&body))?;
    let mut stdout = io::stdout();

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    let mut active_request: Option<u32> = None;

    loop {
        tokio::select! {
            _ = ctrl_c.as_mut() => {
                if let Some(id) = active_request {
                    if let Err(err) = cancel_http_request(host, id).await {
                        eprintln!("\nFailed to cancel request {}: {}", id, err);
                    } else {
                        println!("\n[request {} cancelled]", id);
                    }
                } else {
                    println!("\n[cancelled before request id was assigned]");
                }
                es.close();
                break;
            }
            event = es.next() => {
                match event {
                    Some(Ok(Event::Message(message))) => {
                        let data = message.data;
                        if data.is_empty() {
                            continue;
                        }
                        let token: TokenStream = serde_json::from_str(&data)?;
                        active_request.get_or_insert(token.request_id);
                        stdout.write_all(token.text.as_bytes()).await?;
                        stdout.flush().await?;
                        if token.finished {
                            println!(
                                "\n[request {} finished: {:?}]",
                                token.request_id, token.finish_reason
                            );
                            es.close();
                            break;
                        }
                    }
                    Some(Ok(Event::Open)) => {}
                    Some(Err(err)) => {
                        es.close();
                        anyhow::bail!("stream error: {}", err);
                    }
                    None => break,
                }
            }
        }
    }
    Ok(())
}

async fn fetch_status(host: &str, request_id: u32) -> anyhow::Result<()> {
    let client = Client::new();
    let url = format!("{}/status/{}", host.trim_end_matches('/'), request_id);
    let resp = client.get(url).send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await
            .unwrap_or_else(|e| format!("<failed to read error body: {}>", e));
        anyhow::bail!("Server returned error {}: {}", status, text);
    }
    let status: GenerateResponse = resp.json().await?;
    println!(
        "request_id: {}, finished: {}, reason: {:?}",
        status.request_id, status.finished, status.finish_reason
    );
    println!("text:\n{}", status.text);
    Ok(())
}

async fn cancel_http_request(host: &str, request_id: u32) -> anyhow::Result<()> {
    let client = Client::new();
    let url = format!("{}/cancel/{}", host.trim_end_matches('/'), request_id);
    let resp = client.post(url).send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await
            .unwrap_or_else(|e| format!("<failed to read error body: {}>", e));
        anyhow::bail!("Server returned error {}: {}", status, text);
    }
    let response: GenerateResponse = resp.json().await?;
    println!(
        "Cancelled request {} (finished: {:?})",
        response.request_id, response.finish_reason
    );
    println!("text:\n{}", response.text);
    Ok(())
}

async fn list_models(dir_override: Option<&str>) -> anyhow::Result<()> {
    let models = discover_models(dir_override)?;
    if models.is_empty() {
        let dir = dir_override
            .map(|s| s.to_string())
            .or_else(|| std::env::var("ROCMFORGE_MODELS").ok())
            .unwrap_or_else(|| "models".to_string());
        println!(
            "No GGUF models found in '{}'. Use --dir or ROCMFORGE_MODELS to specify a directory.",
            dir
        );
    } else {
        println!("Discovered models:");
        for model in models {
            let tokenizer = model.tokenizer.as_deref().unwrap_or("(no tokenizer)");
            println!("- {}", model.name);
            println!("  gguf: {}", model.path);
            println!("  tokenizer: {}", tokenizer);
            if let Some(meta) = &model.metadata {
                println!(
                    "  arch: {} | layers: {} | heads: {} | hidden: {} | ctx: {} | vocab: {} | file_type: {} | embedded tokenizer: {}",
                    meta.architecture,
                    meta.num_layers,
                    meta.num_heads,
                    meta.hidden_size,
                    meta.max_position_embeddings,
                    meta.vocab_size,
                    meta.file_type,
                    if meta.has_tokenizer { "yes" } else { "no" }
                );
            }
            if let Some(status) = &model.cache_status {
                println!(
                    "  cache: {}",
                    if status.refreshed {
                        "refreshed"
                    } else if status.cached {
                        "cached"
                    } else {
                        "direct"
                    }
                );
            }
        }
    }
    Ok(())
}

async fn run_local_generate(
    gguf: &str,
    tokenizer: &TokenizerAdapter,
    params: &GenerateRequest,
) -> anyhow::Result<()> {
    let engine = create_engine(gguf).await?;
    let prompt_tokens = tokenizer.encode(&params.prompt);
    let max_tokens = params.max_tokens.unwrap_or(128);
    let request_id = engine
        .submit_request(
            prompt_tokens,
            max_tokens,
            params.temperature.unwrap_or(1.0),
            params.top_k.unwrap_or(50),
            params.top_p.unwrap_or(0.9),
        )
        .await?;

    let mut completion = Box::pin(wait_for_completion(&engine, tokenizer, request_id));
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    let response = tokio::select! {
        res = &mut completion => Some(res?),
        _ = ctrl_c.as_mut() => None,
    };

    if let Some(response) = response {
        println!("request_id: {}", response.request_id);
        println!("finish_reason: {:?}", response.finish_reason);
        println!("text:\n{}", response.text);
    } else {
        engine
            .cancel_request(request_id)
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        if let Some(status) = engine.get_request_status(request_id).await? {
            println!("\n[request {} cancelled]", request_id);
            println!(
                "partial text:\n{}",
                tokenizer.decode(&status.generated_tokens)
            );
        } else {
            println!("\n[request {} cancelled]", request_id);
        }
    }
    engine.stop().await.ok();
    Ok(())
}

async fn run_local_stream(
    gguf: &str,
    tokenizer: &TokenizerAdapter,
    params: &GenerateRequest,
) -> anyhow::Result<()> {
    let engine = create_engine(gguf).await?;
    let prompt_tokens = tokenizer.encode(&params.prompt);
    let max_tokens = params.max_tokens.unwrap_or(128);
    let request_id = engine
        .submit_request(
            prompt_tokens,
            max_tokens,
            params.temperature.unwrap_or(1.0),
            params.top_k.unwrap_or(50),
            params.top_p.unwrap_or(0.9),
        )
        .await?;

    let mut stdout = io::stdout();
    let mut last_idx = 0usize;
    let mut ticker = tokio::time::interval(Duration::from_millis(25));
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            _ = ctrl_c.as_mut() => {
                engine
                    .cancel_request(request_id)
                    .await
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                println!("\n[request {} cancelled]", request_id);
                break;
            }
            _ = ticker.tick() => {
                let status = engine
                    .get_request_status(request_id)
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("request {} disappeared", request_id))?;
                while last_idx < status.generated_tokens.len() {
                    let token = status.generated_tokens[last_idx];
                    stdout
                        .write_all(tokenizer.decode_token(token).as_bytes())
                        .await?;
                    stdout.flush().await?;
                    last_idx += 1;
                }
                if status.is_complete() {
                    println!(
                        "\n[request {} finished: {}]",
                        status.request_id,
                        status.finish_reason.unwrap_or_else(|| "completed".to_string())
                    );
                    println!();
                    break;
                }
            }
        }
    }
    engine.stop().await.ok();
    Ok(())
}

async fn create_engine(gguf: &str) -> anyhow::Result<Arc<InferenceEngine>> {
    let mut engine = InferenceEngine::new(EngineConfig::default())?;
    engine.load_gguf_model(gguf).await?;
    let engine = Arc::new(engine);
    engine.start().await?;

    // Start inference loop in background - don't block on it!
    // Note: run_inference_loop() internally spawns the task, so we don't spawn here
    engine.run_inference_loop().await;

    Ok(engine)
}

async fn wait_for_completion(
    engine: &Arc<InferenceEngine>,
    tokenizer: &TokenizerAdapter,
    request_id: u32,
) -> anyhow::Result<GenerateResponse> {
    loop {
        let status = engine
            .get_request_status(request_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("request {} disappeared", request_id))?;
        if status.is_complete() {
            let text = tokenizer.decode(&status.generated_tokens);
            return Ok(GenerateResponse {
                request_id: status.request_id,
                text,
                tokens: status.generated_tokens.clone(),
                finished: true,
                finish_reason: status
                    .finish_reason
                    .clone()
                    .or(Some("completed".to_string())),
            });
        }
        sleep(Duration::from_millis(25)).await;
    }
}
