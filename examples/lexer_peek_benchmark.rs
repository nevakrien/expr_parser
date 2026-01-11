use expr_parser::parsing::Parser;
use std::time::Instant;

fn build_input(tokens: &[&str], repeat: usize, separator: &str) -> String {
    let mut output = String::new();
    for _ in 0..repeat {
        for token in tokens {
            output.push_str(token);
            output.push_str(separator);
        }
    }
    output
}

fn run_next_only(label: &str, input: &str) -> usize {
    let mut lexer = Parser::new(input, 0);
    let mut token_count = 0;
    loop {
        match lexer.next_token().expect("lexing failed") {
            Some(_) => token_count += 1,
            None => break,
        }
    }
    println!("{label}: next-only -> {token_count} tokens");
    token_count
}

fn run_peek_then_next(label: &str, input: &str) -> usize {
    let mut lexer = Parser::new(input, 0);
    let mut token_count = 0;
    loop {
        let peeked = lexer.peek_token().expect("lexing failed");
        if peeked.is_none() {
            break;
        }
        let _ = lexer.next_token().expect("lexing failed");
        token_count += 1;
    }
    println!("{label}: peek+next -> {token_count} tokens");
    token_count
}

fn benchmark_case(label: &str, input: &str) {
    let start = Instant::now();
    let next_tokens = run_next_only(label, input);
    let next_duration = start.elapsed();

    let start = Instant::now();
    let peek_tokens = run_peek_then_next(label, input);
    let peek_duration = start.elapsed();

    println!("{label}: tokens in input = {next_tokens}");
    println!("{label}: next-only time = {next_duration:?}");
    println!("{label}: peek+next time = {peek_duration:?}");
    println!(
        "{label}: peek/next ratio = {:.2}",
        peek_duration.as_secs_f64() / next_duration.as_secs_f64()
    );
    if next_tokens != peek_tokens {
        println!("{label}: token count mismatch: next={next_tokens}, peek+next={peek_tokens}");
    }
    println!();
}

fn main() {
    println!("Running lexer peek benchmark");

    let keyword_tokens = [
        "let", "const", "type", "struct", "union", "enum", "fn", "cfn", "if", "else", "while",
        "for", "match", "return", "break", "continue", "as",
    ];
    let ident_tokens = [
        "identifier",
        "long_name",
        "foo",
        "bar",
        "baz",
        "alpha1",
        "beta2",
        "gamma3",
    ];
    let operator_tokens = [
        "+", "-", "*", "/", "==", "!=", "<=", ">=", "&&", "||", "->", "::", "<<=", ">>=", "(", ")",
        "{", "}", "[", "]", ";", ",", ".",
    ];
    let numeric_tokens = ["0", "1", "42", "99999", "3.14", "0.001", "1234567890"];
    let string_tokens = ["\"hello\"", "\"world\"", "\"escape\\n\"", "\"tab\\t\""];
    let mixed_tokens = [
        "let", "value", "=", "123", "+", "456", ";", "if", "value", ">", "0", "{", "value", "}",
        "else", "{", "0", "}",
    ];
    let whitespace_separator = "  \n\t  ";

    let keyword_input = build_input(&keyword_tokens, 10000, " ");
    let ident_input = build_input(&ident_tokens, 10000, " ");
    let operator_input = build_input(&operator_tokens, 10000, " ");
    let numeric_input = build_input(&numeric_tokens, 20000, " ");
    let string_input = build_input(&string_tokens, 20000, " ");
    let mixed_input = build_input(&mixed_tokens, 20000, " ");
    let whitespace_input = build_input(&mixed_tokens, 20000, whitespace_separator);

    benchmark_case("keywords", &keyword_input);
    benchmark_case("idents", &ident_input);
    benchmark_case("operators", &operator_input);
    benchmark_case("numbers", &numeric_input);
    benchmark_case("strings", &string_input);
    benchmark_case("mixed", &mixed_input);
    benchmark_case("whitespace", &whitespace_input);
}
