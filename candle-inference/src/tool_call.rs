//! Tool call infrastructure for function-calling models.
//!
//! Provides structured parsing, dispatch, and result injection for
//! LLM-generated tool/function calls.  The module is model-agnostic:
//! any model that emits tool calls in a known text format can be wired
//! through this layer.
//!
//! # Architecture
//!
//! 1. **[`ToolDef`]** — a tool definition (name + parameter schema).
//! 2. **[`ToolCall`]** — a parsed invocation (name + JSON arguments).
//! 3. **[`ToolResult`]** — the output returned by the tool.
//! 4. **[`ToolRegistry`]** — maps tool names to definitions and dispatches
//!    calls to user-provided handler functions.
//!
//! # Example
//!
//! ```rust
//! use candle_inference::tool_call::{ToolDef, ToolCall, ToolResult, ToolRegistry, ParamDef};
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(ToolDef {
//!     name: "get_weather".into(),
//!     description: "Get weather for a city".into(),
//!     parameters: vec![
//!         ParamDef::required("city", "string", "City name"),
//!     ],
//! });
//!
//! // Parse a tool call from model output
//! let call = ToolCall::new("get_weather", r#"{"city":"Paris"}"#);
//! assert!(registry.has_tool(&call.name));
//!
//! // Validate parameters
//! let errors = registry.validate(&call);
//! assert!(errors.is_empty());
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A tool parameter definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDef {
    /// Parameter name.
    pub name: String,
    /// Type hint (e.g., `"string"`, `"integer"`, `"object"`).
    pub type_hint: String,
    /// Human-readable description.
    pub description: String,
    /// Whether this parameter is required.
    pub required: bool,
}

impl ParamDef {
    /// Create a required parameter definition.
    pub fn required(name: impl Into<String>, type_hint: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            type_hint: type_hint.into(),
            description: description.into(),
            required: true,
        }
    }

    /// Create an optional parameter definition.
    pub fn optional(name: impl Into<String>, type_hint: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            type_hint: type_hint.into(),
            description: description.into(),
            required: false,
        }
    }
}

/// A tool definition (schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// Tool name (must be unique within a registry).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Parameter definitions.
    pub parameters: Vec<ParamDef>,
}

impl ToolDef {
    /// Names of required parameters.
    pub fn required_params(&self) -> Vec<&str> {
        self.parameters
            .iter()
            .filter(|p| p.required)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// Get a parameter definition by name.
    pub fn get_param(&self, name: &str) -> Option<&ParamDef> {
        self.parameters.iter().find(|p| p.name == name)
    }
}

/// A parsed tool invocation from model output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name (must match a registered [`ToolDef`]).
    pub name: String,
    /// Arguments as a JSON string.
    pub arguments: String,
    /// Optional call ID for correlating results.
    pub call_id: Option<String>,
}

impl ToolCall {
    /// Create a new tool call.
    pub fn new(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: arguments.into(),
            call_id: None,
        }
    }

    /// Builder: attach a call ID.
    pub fn with_call_id(mut self, id: impl Into<String>) -> Self {
        self.call_id = Some(id.into());
        self
    }

    /// Parse arguments as a serde_json Value.
    ///
    /// Returns `Err` if the arguments string is not valid JSON.
    pub fn parse_arguments(&self) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }
}

/// Result from executing a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Tool name.
    pub name: String,
    /// Call ID (if the originating call had one).
    pub call_id: Option<String>,
    /// Whether the tool succeeded.
    pub success: bool,
    /// Output content (JSON string or plain text).
    pub content: String,
}

impl ToolResult {
    /// Create a successful result.
    pub fn success(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            call_id: None,
            success: true,
            content: content.into(),
        }
    }

    /// Create a failure result.
    pub fn failure(name: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            call_id: None,
            success: false,
            content: error.into(),
        }
    }

    /// Builder: attach a call ID.
    pub fn with_call_id(mut self, id: impl Into<String>) -> Self {
        self.call_id = Some(id.into());
        self
    }

    /// Format as a text block suitable for injection back into the
    /// conversation.
    pub fn format_for_injection(&self) -> String {
        if self.success {
            format!("[Tool Result: {}]\n{}", self.name, self.content)
        } else {
            format!("[Tool Error: {}]\n{}", self.name, self.content)
        }
    }
}

/// A validation error for a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// Tool name not found in registry.
    UnknownTool(String),
    /// Arguments are not valid JSON.
    InvalidJson(String),
    /// A required parameter is missing.
    MissingParam(String),
    /// An argument name does not match any parameter.
    UnknownParam(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownTool(n) => write!(f, "unknown tool: {n}"),
            Self::InvalidJson(e) => write!(f, "invalid JSON: {e}"),
            Self::MissingParam(p) => write!(f, "missing required parameter: {p}"),
            Self::UnknownParam(p) => write!(f, "unknown parameter: {p}"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Registry of available tools.
///
/// Stores [`ToolDef`] entries and provides validation of [`ToolCall`]s.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolDef>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool definition.
    ///
    /// Overwrites any existing tool with the same name.
    pub fn register(&mut self, def: ToolDef) {
        self.tools.insert(def.name.clone(), def);
    }

    /// Remove a tool definition.
    pub fn unregister(&mut self, name: &str) -> Option<ToolDef> {
        self.tools.remove(name)
    }

    /// Check if a tool is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get a tool definition.
    pub fn get(&self, name: &str) -> Option<&ToolDef> {
        self.tools.get(name)
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// True if no tools registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// List all registered tool names.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Validate a tool call against the registry.
    ///
    /// Returns an empty vec on success.
    pub fn validate(&self, call: &ToolCall) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        let def = match self.tools.get(&call.name) {
            Some(d) => d,
            None => {
                errors.push(ValidationError::UnknownTool(call.name.clone()));
                return errors;
            }
        };

        // Parse arguments
        let args = match call.parse_arguments() {
            Ok(v) => v,
            Err(e) => {
                errors.push(ValidationError::InvalidJson(e.to_string()));
                return errors;
            }
        };

        let obj = match args.as_object() {
            Some(o) => o,
            None => {
                errors.push(ValidationError::InvalidJson(
                    "arguments must be a JSON object".into(),
                ));
                return errors;
            }
        };

        // Check required params
        for name in def.required_params() {
            if !obj.contains_key(name) {
                errors.push(ValidationError::MissingParam(name.to_string()));
            }
        }

        // Check for unknown params
        for key in obj.keys() {
            if def.get_param(key).is_none() {
                errors.push(ValidationError::UnknownParam(key.clone()));
            }
        }

        errors
    }

    /// Generate a tool-use system prompt describing all registered tools.
    ///
    /// This can be prepended to the model's context so it knows which tools
    /// are available.
    pub fn system_prompt(&self) -> String {
        let mut parts = Vec::new();
        parts.push("You have access to the following tools:\n".to_string());

        let mut names: Vec<&String> = self.tools.keys().collect();
        names.sort();

        for name in names {
            let def = &self.tools[name];
            parts.push(format!("## {}\n{}\n", def.name, def.description));
            if !def.parameters.is_empty() {
                parts.push("Parameters:\n".to_string());
                for p in &def.parameters {
                    let req = if p.required { "required" } else { "optional" };
                    parts.push(format!("  - {} ({}{}): {}\n", p.name, p.type_hint, 
                        if p.required { "" } else { ", optional" }, p.description));
                    let _ = req; // used in formatting above
                }
            }
        }

        parts.concat()
    }
}

/// Try to extract tool calls from model-generated text.
///
/// Looks for JSON blocks that have `"name"` and `"arguments"` fields.
/// This is a best-effort heuristic; for production use, prefer a
/// model-specific parser.
pub fn extract_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();

    // Look for JSON objects in the text
    for start in text.match_indices('{').map(|(i, _)| i) {
        if let Some(end) = find_matching_brace(text, start) {
            let candidate = &text[start..=end];
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(candidate) {
                if let (Some(name), Some(args)) = (
                    val.get("name").and_then(|v| v.as_str()),
                    val.get("arguments"),
                ) {
                    let args_str = if args.is_string() {
                        args.as_str().unwrap().to_string()
                    } else {
                        args.to_string()
                    };
                    let mut call = ToolCall::new(name, args_str);
                    if let Some(id) = val.get("call_id").and_then(|v| v.as_str()) {
                        call = call.with_call_id(id);
                    }
                    calls.push(call);
                }
            }
        }
    }

    calls
}

/// Find the index of the matching closing brace for an opening brace.
fn find_matching_brace(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if bytes.get(open) != Some(&b'{') {
        return None;
    }

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for i in open..bytes.len() {
        let b = bytes[i];
        if escape_next {
            escape_next = false;
            continue;
        }
        if b == b'\\' && in_string {
            escape_next = true;
            continue;
        }
        if b == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if b == b'{' {
            depth += 1;
        } else if b == b'}' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(ToolDef {
            name: "get_weather".into(),
            description: "Get weather for a city".into(),
            parameters: vec![
                ParamDef::required("city", "string", "City name"),
                ParamDef::optional("units", "string", "celsius or fahrenheit"),
            ],
        });
        reg.register(ToolDef {
            name: "search".into(),
            description: "Search the web".into(),
            parameters: vec![
                ParamDef::required("query", "string", "Search query"),
            ],
        });
        reg
    }

    #[test]
    fn register_and_lookup() {
        let reg = sample_registry();
        assert!(reg.has_tool("get_weather"));
        assert!(reg.has_tool("search"));
        assert!(!reg.has_tool("nonexistent"));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn unregister() {
        let mut reg = sample_registry();
        let removed = reg.unregister("search");
        assert!(removed.is_some());
        assert!(!reg.has_tool("search"));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn validate_valid_call() {
        let reg = sample_registry();
        let call = ToolCall::new("get_weather", r#"{"city":"Paris"}"#);
        assert!(reg.validate(&call).is_empty());
    }

    #[test]
    fn validate_with_optional_param() {
        let reg = sample_registry();
        let call = ToolCall::new("get_weather", r#"{"city":"Paris","units":"celsius"}"#);
        assert!(reg.validate(&call).is_empty());
    }

    #[test]
    fn validate_missing_required() {
        let reg = sample_registry();
        let call = ToolCall::new("get_weather", r#"{"units":"celsius"}"#);
        let errors = reg.validate(&call);
        assert!(errors.contains(&ValidationError::MissingParam("city".into())));
    }

    #[test]
    fn validate_unknown_param() {
        let reg = sample_registry();
        let call = ToolCall::new("get_weather", r#"{"city":"Paris","foo":"bar"}"#);
        let errors = reg.validate(&call);
        assert!(errors.contains(&ValidationError::UnknownParam("foo".into())));
    }

    #[test]
    fn validate_unknown_tool() {
        let reg = sample_registry();
        let call = ToolCall::new("nonexistent", r#"{}"#);
        let errors = reg.validate(&call);
        assert!(errors.contains(&ValidationError::UnknownTool("nonexistent".into())));
    }

    #[test]
    fn validate_invalid_json() {
        let reg = sample_registry();
        let call = ToolCall::new("get_weather", "not json");
        let errors = reg.validate(&call);
        assert!(matches!(errors[0], ValidationError::InvalidJson(_)));
    }

    #[test]
    fn tool_call_with_id() {
        let call = ToolCall::new("test", "{}").with_call_id("abc-123");
        assert_eq!(call.call_id.as_deref(), Some("abc-123"));
    }

    #[test]
    fn tool_result_formatting() {
        let ok = ToolResult::success("weather", "Sunny, 22°C");
        assert!(ok.format_for_injection().contains("[Tool Result: weather]"));

        let err = ToolResult::failure("weather", "API timeout");
        assert!(err.format_for_injection().contains("[Tool Error: weather]"));
    }

    #[test]
    fn extract_tool_calls_basic() {
        let text = r#"I'll check the weather. {"name":"get_weather","arguments":{"city":"Paris"}} Done."#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn extract_tool_calls_with_string_args() {
        let text = r#"{"name":"search","arguments":"{\"query\":\"rust lang\"}"}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "search");
    }

    #[test]
    fn extract_tool_calls_none() {
        let text = "No tool calls here, just regular text.";
        assert!(extract_tool_calls(text).is_empty());
    }

    #[test]
    fn extract_tool_calls_with_call_id() {
        let text = r#"{"name":"test","arguments":{},"call_id":"id-42"}"#;
        let calls = extract_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id.as_deref(), Some("id-42"));
    }

    #[test]
    fn system_prompt_generation() {
        let reg = sample_registry();
        let prompt = reg.system_prompt();
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("search"));
        assert!(prompt.contains("city"));
    }

    #[test]
    fn required_params() {
        let reg = sample_registry();
        let def = reg.get("get_weather").unwrap();
        let req = def.required_params();
        assert_eq!(req, vec!["city"]);
    }

    #[test]
    fn param_def_constructors() {
        let req = ParamDef::required("x", "int", "The x value");
        assert!(req.required);

        let opt = ParamDef::optional("y", "int", "The y value");
        assert!(!opt.required);
    }

    #[test]
    fn empty_registry() {
        let reg = ToolRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.tool_names().is_empty());
    }

    #[test]
    fn find_matching_brace_basic() {
        assert_eq!(find_matching_brace("{}", 0), Some(1));
        assert_eq!(find_matching_brace("{{}}", 0), Some(3));
        assert_eq!(find_matching_brace(r#"{"a":"}"}"#, 0), Some(8));
    }

    #[test]
    fn validation_error_display() {
        let e = ValidationError::UnknownTool("foo".into());
        assert_eq!(e.to_string(), "unknown tool: foo");

        let e = ValidationError::MissingParam("bar".into());
        assert_eq!(e.to_string(), "missing required parameter: bar");
    }
}
