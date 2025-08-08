# {py:mod}`data.hf_datasets.chat_templates`

```{py:module} data.hf_datasets.chat_templates
```

```{autodoc2-docstring} data.hf_datasets.chat_templates
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`COMMON_CHAT_TEMPLATES <data.hf_datasets.chat_templates.COMMON_CHAT_TEMPLATES>`
  - ```{autodoc2-docstring} data.hf_datasets.chat_templates.COMMON_CHAT_TEMPLATES
    :summary:
    ```
````

### API

`````{py:class} COMMON_CHAT_TEMPLATES
:canonical: data.hf_datasets.chat_templates.COMMON_CHAT_TEMPLATES

```{autodoc2-docstring} data.hf_datasets.chat_templates.COMMON_CHAT_TEMPLATES
```

````{py:attribute} simple_role_header
:canonical: data.hf_datasets.chat_templates.COMMON_CHAT_TEMPLATES.simple_role_header
:value: <Multiline-String>

```{autodoc2-docstring} data.hf_datasets.chat_templates.COMMON_CHAT_TEMPLATES.simple_role_header
```

````

````{py:attribute} passthrough_prompt_response
:canonical: data.hf_datasets.chat_templates.COMMON_CHAT_TEMPLATES.passthrough_prompt_response
:value: >
   "{% for message in messages %}{{ message['content'] }}{% endfor %}"

```{autodoc2-docstring} data.hf_datasets.chat_templates.COMMON_CHAT_TEMPLATES.passthrough_prompt_response
```

````

`````
