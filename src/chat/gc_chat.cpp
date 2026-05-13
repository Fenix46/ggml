#include "gc_chat.h"
#include "gc_debug.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <map>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
//  TOKEN TYPES
// ═══════════════════════════════════════════════════════════════════════════════

enum class TokenType {
    Text,               // raw text between delimiters
    VariableOpen,       // {{
    VariableClose,      // }}
    TagOpen,            // {%
    TagClose,           // %}
    CommentOpen,        // {#
    CommentClose,       // #}
    Identifier,         // a-z A-Z _ . 0-9
    Number,             // int or float
    String,             // "..." or '...'
    Op,                 // == != < > <= >=
    Comma,              // ,
    Dot,                // .
    Pipe,               // |
    Assign,             // =
    Eof,
};

struct Token {
    TokenType  type;
    std::string value;
    size_t     pos;    // source position for error messages
};

// ═══════════════════════════════════════════════════════════════════════════════
//  AST NODES
// ═══════════════════════════════════════════════════════════════════════════════

struct AstNode {
    enum Kind {
        kRoot,
        kText,
        kExpr,       // {{ expr }}
        kIf,
        kElse,
        kEndIf,
        kFor,
        kEndFor,
        kSet,
    };
    Kind kind;

    // kText
    std::string text;

    // kExpr: variable path + filter chain
    std::string expr_var;        // e.g. "messages[0].content"
    std::vector<std::string> filters;  // e.g. strip, upper

    // kIf / kElse
    std::string cond_var;        // variable name for condition
    bool        cond_truth;      // computed at render time

    // kFor
    std::string loop_var;        // e.g. "item"
    std::string loop_iter;       // e.g. "messages"

    // kSet
    std::string set_var;
    std::string set_expr;

    // Children
    std::vector<AstNode> children;
};

struct AstRoot {
    std::vector<AstNode> nodes;
};

// ═══════════════════════════════════════════════════════════════════════════════
//  LEXER
// ═══════════════════════════════════════════════════════════════════════════════

class Lexer {
public:
    explicit Lexer(const std::string & src)
        : src_(src), pos_(0), line_(1), col_(0) {}

    std::vector<Token> tokenize();

private:
    const std::string & src_;
    size_t             pos_;
    int                line_;
    int                col_;

    char peek()  const { return pos_ < src_.size() ? src_[pos_] : '\0'; }
    char adv()         { char c = src_[pos_++]; col_++; if (c == '\n') { line_++; col_=0; } return c; }
    void skip_ws()     { while (peek() && (peek() == ' ' || peek() == '\t' || peek() == '\n')) adv(); }

    Token make(TokenType t, std::string v = "") {
        return Token{t, std::move(v), pos_};
    }

    Token make(TokenType t, const char * s, size_t len) {
        return Token{t, std::string(s, len), pos_};
    }

    void error(const std::string & msg);
};

std::vector<Token> Lexer::tokenize() {
    std::vector<Token> tokens;
    size_t text_start = 0;
    bool   in_text    = true;

    auto emit_text = [&]() {
        if (pos_ > text_start) {
            tokens.push_back(make(TokenType::Text, &src_[text_start], pos_ - text_start));
        }
    };

    while (pos_ < src_.size()) {
        if (peek() == '{' && pos_ + 1 < src_.size()) {
            char n = src_[pos_ + 1];
            if (n == '{') {  // {{
                emit_text();
                adv(); adv(); // {{ consumed
                tokens.push_back(make(TokenType::VariableOpen));
                in_text = false;
                // Parse expression inside {{ }}
                while (pos_ < src_.size()) {
                    skip_ws();
                    if (peek() == '}' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '}') {
                        emit_text();  // will be empty
                        adv(); adv();
                        tokens.push_back(make(TokenType::VariableClose));
                        in_text = true;
                        text_start = pos_;
                        break;
                    }
                    // Parse identifiers / operators / string literals
                    if (isalpha(peek()) || peek() == '_') {
                        size_t s = pos_;
                        while (pos_ < src_.size() && (isalnum(peek()) || peek() == '_' || peek() == '.' || peek() == '[' || peek() == ']' || peek() == '0' || peek() == '1' || peek() == '2' || peek() == '3' || peek() == '4' || peek() == '5' || peek() == '6' || peek() == '7' || peek() == '8' || peek() == '9')) {
                            // Check for [0-9] for indices like messages[0]
                            if (peek() == '[') { adv(); continue; }
                            if (peek() == ']') { adv(); continue; }
                            if (peek() == '.') { adv(); continue; }
                            if (isdigit(peek())) {
                                // Check if we're inside brackets -- that's an index
                                size_t back = pos_;
                                adv();
                                continue;
                            }
                            adv();
                        }
                        tokens.push_back(make(TokenType::Identifier, &src_[s], pos_ - s));
                        skip_ws();
                    } else if (peek() == '"' || peek() == '\'') {
                        size_t s = pos_;
                        char quote = adv();
                        while (pos_ < src_.size() && peek() != quote) adv();
                        adv();
                        tokens.push_back(make(TokenType::String, &src_[s], pos_ - s));
                        skip_ws();
                        // Actually the string includes delimiters, which we need to strip later
                    } else if (peek() == '|') {
                        adv();
                        tokens.push_back(make(TokenType::Pipe));
                        skip_ws();
                    } else if (peek() == ',') {
                        adv();
                        tokens.push_back(make(TokenType::Comma));
                        skip_ws();
                    } else if (peek() == '=' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '=') {
                        adv(); adv();
                        tokens.push_back(make(TokenType::Op, "=="));
                        skip_ws();
                    } else if (peek() == '!' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '=') {
                        adv(); adv();
                        tokens.push_back(make(TokenType::Op, "!="));
                        skip_ws();
                    } else if (peek() == '<' || peek() == '>') {
                        char c = adv();
                        if (peek() == '=') { adv(); tokens.push_back(make(TokenType::Op, std::string(1, c) + "=")); }
                        else { tokens.push_back(make(TokenType::Op, std::string(1, c))); }
                        skip_ws();
                    } else if (peek() == '-' && pos_ + 1 < src_.size() && isdigit(src_[pos_ + 1])) {
                        size_t s = pos_;
                        adv(); // consume -
                        while (pos_ < src_.size() && isdigit(peek())) adv();
                        tokens.push_back(make(TokenType::Number, &src_[s], pos_ - s));
                        skip_ws();
                    } else if (isdigit(peek())) {
                        size_t s = pos_;
                        while (pos_ < src_.size() && isdigit(peek())) adv();
                        tokens.push_back(make(TokenType::Number, &src_[s], pos_ - s));
                        skip_ws();
                    } else if (peek() == '(') {
                        // skip parens for function calls — we only support simple variable lookups
                        adv();
                        skip_ws();
                    } else if (peek() == ')') {
                        adv();
                        skip_ws();
                    } else {
                        break;
                    }
                }
                continue;
            } else if (n == '%') {  // {%
                emit_text();
                adv(); adv();
                tokens.push_back(make(TokenType::TagOpen));
                in_text = false;
                // Parse tag content
                while (pos_ < src_.size()) {
                    skip_ws();
                    if (peek() == '%' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '}') {
                        emit_text();
                        adv(); adv();
                        tokens.push_back(make(TokenType::TagClose));
                        in_text = true;
                        text_start = pos_;
                        break;
                    }
                    if (isalpha(peek()) || peek() == '_') {
                        size_t s = pos_;
                        while (pos_ < src_.size() && (isalnum(peek()) || peek() == '_' || peek() == '.' || peek() == '[' || peek() == ']')) adv();
                        tokens.push_back(make(TokenType::Identifier, &src_[s], pos_ - s));
                        skip_ws();
                    } else if (peek() == '"' || peek() == '\'') {
                        size_t s = pos_;
                        char quote = adv();
                        while (pos_ < src_.size() && peek() != quote) adv();
                        adv();
                        tokens.push_back(make(TokenType::String, &src_[s], pos_ - s));
                        skip_ws();
                    } else if (peek() == '=' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '=') {
                        adv(); adv();
                        tokens.push_back(make(TokenType::Op, "=="));
                        skip_ws();
                    } else if (peek() == '!' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '=') {
                        adv(); adv();
                        tokens.push_back(make(TokenType::Op, "!="));
                        skip_ws();
                    } else if (peek() == '<' || peek() == '>') {
                        char c = adv();
                        if (peek() == '=') { adv(); tokens.push_back(make(TokenType::Op, std::string(1, c) + "=")); }
                        else { tokens.push_back(make(TokenType::Op, std::string(1, c))); }
                        skip_ws();
                    } else if (peek() == ',') { adv(); tokens.push_back(make(TokenType::Comma)); skip_ws(); }
                    else break;
                }
                continue;
            } else if (n == '#') {  // {#
                emit_text();
                adv(); adv();
                tokens.push_back(make(TokenType::CommentOpen));
                in_text = false;
                while (pos_ < src_.size()) {
                    if (peek() == '#' && pos_ + 1 < src_.size() && src_[pos_ + 1] == '}') {
                        adv(); adv();
                        tokens.push_back(make(TokenType::CommentClose));
                        in_text = true;
                        text_start = pos_;
                        break;
                    }
                    adv();
                }
                continue;
            }
        }
        adv();
    }
    if (in_text) {
        emit_text();
    }
    tokens.push_back(make(TokenType::Eof));
    return tokens;
}

void Lexer::error(const std::string & msg) {
    GC_LOG_ERR("Lexer error at line %d: %s", line_, msg.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PARSER
// ═══════════════════════════════════════════════════════════════════════════════

class Parser {
public:
    Parser(const std::vector<Token> & tokens, std::string & err)
        : tokens_(tokens), pos_(0), err_(err) {}

    AstRoot parse();

private:
    const std::vector<Token> & tokens_;
    size_t                    pos_;
    std::string &             err_;

    const Token & peek() const { return pos_ < tokens_.size() ? tokens_[pos_] : tokens_.back(); }
    const Token & adv()        { return tokens_[pos_++]; }

    bool accept(TokenType t) {
        if (peek().type == t) { adv(); return true; }
        return false;
    }

    bool expect(TokenType t, const std::string & ctx) {
        if (peek().type != t) {
            err_ = std::string("Expected ") + std::to_string((int)t) + " at pos " + std::to_string(peek().pos) + " in " + ctx;
            return false;
        }
        adv();
        return true;
    }

    void skip_eof() {
        while (pos_ < tokens_.size() && peek().type != TokenType::Eof) {
            if (peek().type == TokenType::Text) { adv(); continue; }
            if (peek().type == TokenType::CommentClose) { adv(); continue; }
            break;
        }
        adv(); // Eof
    }
};

AstRoot Parser::parse() {
    AstRoot root;
    std::vector<AstNode> * current = &root.nodes;
    std::stack<std::vector<AstNode> *> stack;
    stack.push(current);

    auto err_if = [&](bool cond, const std::string & msg) {
        if (cond) { err_ = msg; }
    };

    while (pos_ < tokens_.size()) {
        const Token & t = peek();
        if (t.type == TokenType::Eof) break;

        if (t.type == TokenType::Text) {
            AstNode n;
            n.kind  = AstNode::kText;
            n.text  = t.value;
            stack.top()->push_back(n);
            adv();
            continue;
        }

        if (t.type == TokenType::CommentOpen) {
            while (pos_ < tokens_.size() && peek().type != TokenType::CommentClose) adv();
            if (pos_ < tokens_.size()) adv(); // consume CommentClose
            continue;
        }

        if (t.type == TokenType::VariableOpen) {
            adv(); // consume {{
            AstNode n;
            n.kind = AstNode::kExpr;

            // Parse variable expression
            std::string var;
            while (pos_ < tokens_.size()) {
                const Token & in = peek();
                if (in.type == TokenType::Identifier) {
                    var += in.value;
                    adv();
                } else if (in.type == TokenType::Pipe) {
                    // Start filter chain
                    adv();
                    // Collect filters
                    while (pos_ < tokens_.size() && peek().type == TokenType::Identifier) {
                        n.filters.push_back(peek().value);
                        adv();
                        // Check for next pipe
                        if (pos_ < tokens_.size() && peek().type == TokenType::Pipe) {
                            adv();
                        } else {
                            break;
                        }
                    }
                    // After filters, expect VariableClose
                    if (pos_ < tokens_.size() && peek().type == TokenType::VariableClose) {
                        adv();
                        break;
                    }
                } else if (in.type == TokenType::VariableClose) {
                    adv();
                    break;
                } else {
                    break;
                }
            }
            n.expr_var = var;
            stack.top()->push_back(n);
            continue;
        }

        if (t.type == TokenType::TagOpen) {
            adv(); // consume {%
            // Read the keyword: if, for, else, endif, endfor, set
            std::string keyword;
            if (pos_ < tokens_.size() && peek().type == TokenType::Identifier) {
                keyword = peek().value;
                adv();
            }

            if (keyword == "if") {
                AstNode n;
                n.kind = AstNode::kIf;
                // Read condition variable
                if (pos_ < tokens_.size() && tokens_[pos_].type == TokenType::Identifier) {
                    n.cond_var = tokens_[pos_].value;
                    adv();
                }
                // Read comparison operator and value if present
                if (pos_ < tokens_.size() && tokens_[pos_].type == TokenType::Op) {
                    // For now, simplified: if var is truthy/falsy
                }
                // Expect TagClose
                if (pos_ < tokens_.size() && peek().type == TokenType::TagClose) {
                    adv();
                }
                // Push if block
                stack.top()->push_back(n);
                // Create a new level for if body
                std::vector<AstNode> body;
                stack.push(&stack.top()->back().children);
                continue;
            }

            if (keyword == "else") {
                // Pop current level, create else branch
                stack.pop();
                // Add else node to parent (the if)
                AstNode else_node;
                else_node.kind = AstNode::kElse;
                // We need to find the if node in the parent and add else there
                if (!stack.empty()) {
                    auto & parent = *stack.top();
                    if (!parent.empty() && parent.back().kind == AstNode::kIf) {
                        parent.back().children.push_back(else_node);
                        // Push else body level
                        std::vector<AstNode> else_body;
                        stack.push(&parent.back().children);
                    }
                }
                // Expect TagClose
                if (pos_ < tokens_.size() && peek().type == TokenType::TagClose) adv();
                continue;
            }

            if (keyword == "endif") {
                // Pop if body level
                if (stack.size() > 1) stack.pop();
                if (pos_ < tokens_.size() && peek().type == TokenType::TagClose) adv();
                continue;
            }

            if (keyword == "for") {
                AstNode n;
                n.kind = AstNode::kFor;
                // Read loop_var
                if (pos_ < tokens_.size() && tokens_[pos_].type == TokenType::Identifier) {
                    n.loop_var = tokens_[pos_].value;
                    adv();
                }
                // Expect comma or "in"
                // Skip to the "in" keyword
                while (pos_ < tokens_.size() && peek().type != TokenType::Identifier) adv();
                // After "in", the iterable
                if (pos_ < tokens_.size() && tokens_[pos_].type == TokenType::Identifier) {
                    // Could be "messages" or similar
                    // For now, look for identifier that is the collection
                    if (tokens_[pos_].value != "for" && tokens_[pos_].value != "in") {
                        // It's the iterable name but we keep it in the node
                        n.loop_iter = tokens_[pos_].value;
                        adv();
                    }
                }
                // Check for comma-separated parts (messages[0].role etc.)
                while (pos_ < tokens_.size() && peek().type == TokenType::Comma) {
                    adv();
                    // Skip extra identifiers
                    while (pos_ < tokens_.size() && peek().type == TokenType::Identifier) adv();
                }
                // Expect TagClose
                if (pos_ < tokens_.size() && peek().type == TokenType::TagClose) adv();
                // Push for body level
                stack.top()->push_back(n);
                std::vector<AstNode> body;
                stack.push(&stack.top()->back().children);
                continue;
            }

            if (keyword == "endfor") {
                if (stack.size() > 1) stack.pop();
                if (pos_ < tokens_.size() && peek().type == TokenType::TagClose) adv();
                continue;
            }

            if (keyword == "set") {
                AstNode n;
                n.kind = AstNode::kSet;
                // Read variable name
                if (pos_ < tokens_.size() && tokens_[pos_].type == TokenType::Identifier) {
                    n.set_var = tokens_[pos_].value;
                    adv();
                }
                // Expect =
                if (pos_ < tokens_.size() && peek().type == TokenType::Op && peek().value == "=") {
                    adv();
                }
                // Read expression
                std::string expr;
                while (pos_ < tokens_.size() && peek().type != TokenType::TagClose) {
                    expr += peek().value + " ";
                    adv();
                }
                n.set_expr = expr;
                if (pos_ < tokens_.size() && peek().type == TokenType::TagClose) adv();
                stack.top()->push_back(n);
                continue;
            }

            // Unknown tag — fail loudly per spec
            err_ = "Unknown template tag at position " + std::to_string(t.pos);
            continue;
        }

        // Unknown token — skip
        adv();
    }

    return root;
}

// ═══════════════════════════════════════════════════════════════════════════════
//  RENDERER (EVALUATOR)
// ═══════════════════════════════════════════════════════════════════════════════

struct ChatContext {
    // messages array
    std::vector<std::string> roles;
    std::vector<std::string> contents;

    // System message
    std::string system;

    // Loop iteration state
    struct LoopVar {
        std::string value;
        int         index = 0;
        bool        first = false;
        bool        last  = false;
    };
    std::map<std::string, LoopVar> loop_vars;

    // Set variables
    std::map<std::string, std::string> vars;

    // Get a value by path expression (simplified: only var name or messages[i].field)
    std::string resolve(const std::string & expr) const {
        // Simple variable
        auto it = vars.find(expr);
        if (it != vars.end()) return it->second;

        // Loop variable
        auto lit = loop_vars.find(expr);
        if (lit != loop_vars.end()) return lit->second.value;

        // messages[N].role or messages[N].content
        if (expr.find("messages[") == 0) {
            size_t brk = expr.find(']');
            if (brk != std::string::npos) {
                std::string idx_str = expr.substr(9, brk - 9);
                int idx = 0;
                try { idx = std::stoi(idx_str); } catch (...) {}
                std::string field = expr.substr(brk + 2); // skip ]. and field
                if (idx >= 0 && (size_t)idx < roles.size()) {
                    if (field == "role") return roles[idx];
                    if (field == "content") return contents[idx];
                }
            }
        }

        // messages[N] (whole message as string)
        if (expr.find("messages[") == 0) {
            size_t brk = expr.find(']');
            if (brk != std::string::npos) {
                std::string idx_str = expr.substr(9, brk - 9);
                int idx = 0;
                try { idx = std::stoi(idx_str); } catch (...) {}
                if (idx >= 0 && (size_t)idx < roles.size()) {
                    return roles[idx] + ": " + contents[idx];
                }
            }
        }

        // "messages" as an array — not supported, just return count
        if (expr == "messages") return std::to_string(roles.size());

        // "system"
        if (expr == "system") return system;

        return "";
    }
};

// Apply filters
static std::string apply_filter(const std::string & value, const std::string & filter) {
    if (filter == "strip") {
        size_t s = 0, e = value.size();
        while (s < e && isspace((unsigned char)value[s])) s++;
        while (e > s && isspace((unsigned char)value[e-1])) e--;
        return value.substr(s, e - s);
    }
    if (filter == "upper") {
        std::string r = value;
        for (char & c : r) c = toupper((unsigned char)c);
        return r;
    }
    if (filter == "lower") {
        std::string r = value;
        for (char & c : r) c = tolower((unsigned char)c);
        return r;
    }
    if (filter == "title") {
        std::string r = value;
        bool cap = true;
        for (char & c : r) {
            if (isspace((unsigned char)c)) { cap = true; continue; }
            if (cap) c = toupper((unsigned char)c); cap = false;
        }
        return r;
    }
    if (filter == "length") {
        return std::to_string(value.size());
    }
    // Unknown filter — return as-is
    return value;
}

// Render an AST node to output string
static void render_node(const AstNode & node, ChatContext & ctx, std::string & out) {
    switch (node.kind) {
        case AstNode::kText:
            out += node.text;
            break;

        case AstNode::kExpr: {
            std::string val = ctx.resolve(node.expr_var);
            for (const auto & f : node.filters) {
                val = apply_filter(val, f);
            }
            out += val;
            break;
        }

        case AstNode::kIf: {
            // Evaluate condition: variable truthiness
            std::string cond_val = ctx.resolve(node.cond_var);
            bool truthy = !cond_val.empty() && cond_val != "0" && cond_val != "false";
            // Check if there's an else branch
            bool has_else = false;
            size_t else_idx = 0;
            for (size_t i = 0; i < node.children.size(); i++) {
                if (node.children[i].kind == AstNode::kElse) {
                    has_else = true;
                    else_idx = i;
                    break;
                }
            }
            if (truthy) {
                // Render children up to else
                for (size_t i = 0; i < node.children.size(); i++) {
                    if (node.children[i].kind == AstNode::kElse) break;
                    render_node(node.children[i], ctx, out);
                }
            } else if (has_else) {
                // Render after else
                for (size_t i = else_idx + 1; i < node.children.size(); i++) {
                    render_node(node.children[i], ctx, out);
                }
            }
            break;
        }

        case AstNode::kElse:
            // Handled by kIf above
            break;

        case AstNode::kFor: {
            // Iterate over "messages" or provided collection
            std::string collection = ctx.resolve(node.loop_iter);
            // If the iterable is "messages", iterate over roles/contents
            if (node.loop_iter == "messages") {
                for (size_t i = 0; i < ctx.roles.size(); i++) {
                    ChatContext::LoopVar lv;
                    lv.value  = ctx.roles[i] + ": " + ctx.contents[i];
                    lv.index  = i;
                    lv.first  = (i == 0);
                    lv.last   = (i == ctx.roles.size() - 1);
                    ctx.loop_vars[node.loop_var] = lv;

                    // Also populate "role" and "content" loop vars
                    ChatContext::LoopVar role_lv, content_lv;
                    role_lv.value     = ctx.roles[i];
                    role_lv.index     = i;
                    role_lv.first     = (i == 0);
                    role_lv.last      = (i == ctx.roles.size() - 1);
                    content_lv.value  = ctx.contents[i];
                    content_lv.index  = i;
                    content_lv.first  = (i == 0);
                    content_lv.last   = (i == ctx.roles.size() - 1);
                    ctx.loop_vars[node.loop_var + ".role"] = role_lv;
                    ctx.loop_vars[node.loop_var + ".content"] = content_lv;

                    for (const auto & child : node.children) {
                        render_node(child, ctx, out);
                    }
                }
            }
            break;
        }

        case AstNode::kSet: {
            std::string val = ctx.resolve(node.set_expr);
            ctx.vars[node.set_var] = val;
            break;
        }

        default:
            break;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

struct gc_chat_template_t {
    AstRoot            ast;
    std::string        compile_error;
    std::string        raw_template;
};

gc_status_t gc_chat_create(gc_chat_template_t ** out) {
    *out = new gc_chat_template_t{};
    return *out ? GC_OK : GC_ERR_ALLOC;
}

void gc_chat_free(gc_chat_template_t * t) {
    delete t;
}

gc_status_t gc_chat_compile(gc_chat_template_t * t, const char * template_str) {
    if (!t || !template_str) return GC_ERR_INVALID;
    t->raw_template = template_str;
    t->compile_error.clear();

    Lexer lexer(template_str);
    auto tokens = lexer.tokenize();

    t->compile_error.clear();
    Parser parser(tokens, t->compile_error);
    t->ast = parser.parse();

    if (!t->compile_error.empty()) {
        return GC_ERR_UNSUPPORTED;
    }
    return GC_OK;
}

gc_status_t gc_chat_render(gc_chat_template_t * t,
                           const char * const * roles,
                           const char * const * contents,
                           size_t n_messages,
                           const char * system,
                           bool add_ass,
                           std::string * out) {
    if (!t || !out) return GC_ERR_INVALID;

    ChatContext ctx;
    for (size_t i = 0; i < n_messages; i++) {
        ctx.roles.push_back(roles[i] ? roles[i] : "");
        ctx.contents.push_back(contents[i] ? contents[i] : "");
    }
    ctx.system = system ? system : "";

    out->clear();
    for (const auto & node : t->ast.nodes) {
        render_node(node, ctx, *out);
    }

    // If add_ass, append assistant start
    if (add_ass) {
        // Detect expected assistant prefix from template (simple heuristic)
        if (t->raw_template.find("<|im_start|>assistant") != std::string::npos)
            *out += "<|im_start|>assistant\n";
        else if (t->raw_template.find("<|start_header_id|>assistant") != std::string::npos)
            *out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        else if (t->raw_template.find("[INST]") != std::string::npos && t->raw_template.find("[/INST]") != std::string::npos)
            *out += " [/INST]";
        else if (t->raw_template.find("ASSISTANT:") != std::string::npos)
            *out += "ASSISTANT:";
        else if (t->raw_template.find("model") != std::string::npos && t->raw_template.find("<|end_of_turn|>") != std::string::npos)
            *out += "<|start_of_turn|>model\n";
        else
            *out += "assistant\n";
    }

    return GC_OK;
}

gc_status_t gc_chat_apply(const char * template_str,
                          const char * const * roles,
                          const char * const * contents,
                          size_t n_messages,
                          std::string * out) {
    // Use a default ChatML template if none provided
    if (!template_str) template_str = "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content | strip }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n";

    gc_chat_template_t * t = nullptr;
    gc_status_t st = gc_chat_create(&t);
    if (st != GC_OK) return st;

    st = gc_chat_compile(t, template_str);
    if (st != GC_OK) { gc_chat_free(t); return st; }

    st = gc_chat_render(t, roles, contents, n_messages, "", false, out);
    gc_chat_free(t);
    return st;
}

const char * gc_chat_error(gc_chat_template_t * t) {
    return t ? t->compile_error.c_str() : "null template";
}
