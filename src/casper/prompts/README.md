# Prompts Directory

All LLM prompts for CASPER components.

## File Naming Convention

- **`*_system.txt`**: System prompts (static, no variables)
- **`*_user.jinja`**: User prompts (templates with Jinja2 variables)

## Templates

### Preference Extraction
- `preference_extraction_system.txt` - System role definition
- `preference_extraction_user.jinja` - User prompt with variables:
  - `{{ conversation_turns }}` - Full conversation history
  - `{{ existing_preferences }}` - Previously extracted preferences dict

### Question Generation
- `question_generation_system.txt` - System role definition
- `question_generation_user.jinja` - User prompt with variables:
  - `{{ entities }}` - List of RL-selected entities
  - `{{ conversation_context }}` - Full conversation history
  - `{{ discovered_preferences }}` - Dict of extracted preferences

## Jinja2 Syntax

Variable injection: `{{ variable_name }}`
Conditionals: `{% if variable %} ... {% endif %}`
Loops: `{% for item in list %} ... {% endfor %}`
Filters: `{{ list | join(', ') }}`, `{{ dict | tojson }}`
