use rocmforge::loader::{GgufFile, TokenizerData};
use rocmforge::tokenizer::SpmTokenizer;

#[test]
fn encode_paris_is_the() {
    let path = "/home/feanor/Projects/models/gemma-4-e2b-ollama.gguf";
    let file = GgufFile::open(path).expect("open ollama gemma4 gguf");
    let data: &TokenizerData = file.tokenizer_data();
    println!("model={:?} pre={:?}", data.model, data.pre);
    println!(
        "tokens len={} scores len={} types len={}",
        data.tokens.len(),
        data.scores.len(),
        data.token_types.len()
    );
    // print sample tokens
    for tid in [0, 1, 2, 3, 50429, 563, 506, 9079] {
        if let Some(t) = data.tokens.get(tid) {
            println!(
                "token {} {:?} -> {:?}",
                tid,
                String::from_utf8_lossy(t),
                std::str::from_utf8(t)
            );
        }
    }
    let tok = SpmTokenizer::from_gguf(data);
    let ids = tok.encode("Paris is the", false);
    println!("encoded {:?}", ids);
    for &id in &ids {
        println!("  id {} {:?}", id, tok.decode(&[id], false));
    }
    assert_eq!(ids, vec![50429, 563, 506]);

    let cases = [
        ("Paris is the", vec![50429, 563, 506]),
        (" Paris is the", vec![9079, 563, 506]),
        (
            "The quick brown fox jumps over the lazy dog",
            vec![818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799],
        ),
        ("Hello, world!", vec![9259, 236764, 1902, 236888]),
        ("<bos>Hello", vec![2, 9259]),
    ];
    for (text, expected) in cases {
        let ids = tok.encode(text, false);
        println!("{:?} -> {:?}", text, ids);
        assert_eq!(ids, expected, "mismatch for {:?}", text);
    }
}
