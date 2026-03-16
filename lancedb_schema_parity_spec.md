# LanceDB Schema Parity Spec (Phase 2)

This spec is generated from `schema/lancedb_schema_contract.json` and enforced by CI.

## Source-specific constraints

- `dailymed`: `57` fields, `3` sampled payloads
- `pmc_author_manuscript`: `42` fields, `2` sampled payloads
- `pmc_oa`: `42` fields, `2` sampled payloads
- `pubmed_abstract`: `54` fields, `2` sampled payloads

## Vector Expectations

- Expected vector dimension: `1024`
- Runtime vector dimension: `1024`
- Manifest vector column: `vector`

## Field-by-field parity table

| Field | dailymed | pmc_author_manuscript | pmc_oa | pubmed_abstract |
|---|---|---|---|---|
| abstract | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| abstract_structured | - | - | - | `list<dict{label:str,text:str}>, list<empty>` (required, non-null) |
| active_ingredients | `list<str>` (required, non-null) | - | - | - |
| article_type | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| chunk_id | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| chunk_index | `int` (required, non-null) | `int` (required, non-null) | `int` (required, non-null) | `int` (required, non-null) |
| content_type | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| country | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| doc_id | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| doi | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `null, str` (required, nullable) |
| drug_name | `str` (required, non-null) | - | - | - |
| evidence_grade | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| evidence_level | `null` (required, nullable) | `int` (required, non-null) | `int` (required, non-null) | `int` (required, non-null) |
| evidence_source | `null` (required, nullable) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| evidence_term | `null` (required, nullable) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| full_section_text | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| full_text | `str` (required, non-null) | - | - | `str` (required, non-null) |
| gov_agencies | - | - | - | `list<empty>, list<str>` (required, non-null) |
| has_full_text | `bool` (required, non-null) | `bool` (required, non-null) | `bool` (required, non-null) | `bool` (required, non-null) |
| has_structured_abstract | - | - | - | `bool` (required, non-null) |
| has_tables | `bool` (required, non-null) | - | - | - |
| ingestion_timestamp | `float` (required, non-null) | - | - | - |
| is_author_manuscript | `bool` (required, non-null) | `bool` (required, non-null) | `bool` (required, non-null) | `bool` (required, non-null) |
| is_gov_affiliated | - | - | - | `bool` (required, non-null) |
| is_open_access | `null` (required, nullable) | `bool` (required, non-null) | `bool` (required, non-null) | `null` (required, nullable) |
| is_table | `bool` (required, non-null) | `bool` (required, non-null) | `bool` (required, non-null) | - |
| journal | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| journal_full | - | - | - | `dict{abbreviation:str,issn_electronic:str,issn_print:str,issue:str,nlm_unique_id:str,publisher:str,title:str,volume:str}, dict{}` (required, non-null) |
| keywords | `list<empty>` (required, non-null) | `list<str>` (required, non-null) | `list<str>` (required, non-null) | `list<empty>, list<str>` (required, non-null) |
| keywords_full | - | - | - | `list<dict{keyword:str,major_yn:str}>, list<empty>` (required, non-null) |
| label_type_code | `str` (required, non-null) | - | - | - |
| label_type_display | `str` (required, non-null) | - | - | - |
| license | - | `str` (required, non-null) | `str` (required, non-null) | - |
| manufacturer | `str` (required, non-null) | - | - | - |
| mesh_terms | `list<empty>` (required, non-null) | `list<str>` (required, non-null) | `list<str>` (required, non-null) | `list<empty>, list<str>` (required, non-null) |
| mesh_terms_full | - | - | - | `list<dict{descriptor:str,qualifiers:list<str>}>, list<empty>` (required, non-null) |
| nihms_id | `null` (required, nullable) | `str` (required, non-null) | `null` (required, nullable) | `null` (required, nullable) |
| nlm_unique_id | `null` (required, nullable) | `str` (required, non-null) | `str` (required, non-null) | `null, str` (required, nullable) |
| other_ids | `dict{}` (required, non-null) | - | - | `dict{mid:str,pmcpid:str}, dict{}` (required, non-null) |
| page_content | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| parent_section_id | `null` (required, nullable) | - | - | `null` (required, nullable) |
| pii | `null` (required, nullable) | - | - | `null, str` (required, nullable) |
| pmcid | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `null, str` (required, nullable) |
| pmid | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| point_id | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| publication_date | - | - | - | `dict{day:int,month:int,year:int}, dict{}` (required, non-null) |
| publication_type | `list<str>` (required, non-null) | `list<str>` (required, non-null) | `list<str>` (required, non-null) | `list<empty>, list<str>` (required, non-null) |
| publication_types_full | - | - | - | `list<dict{type:str,ui:str}>, list<empty>` (required, non-null) |
| section_id | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| section_title | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| section_type | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| section_weight | `float` (required, non-null) | `float` (required, non-null) | `float` (required, non-null) | `float` (required, non-null) |
| set_id | `str` (required, non-null) | - | - | - |
| source | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| source_family | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| sparse_indices | `list<int>` (required, non-null) | `list<int>` (required, non-null) | `list<int>` (required, non-null) | `list<int>` (required, non-null) |
| sparse_values | `list<float>` (required, non-null) | `list<float>` (required, non-null) | `list<float>` (required, non-null) | `list<float>` (required, non-null) |
| table_caption | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | - |
| table_count | `int` (required, non-null) | - | - | `int` (required, non-null) |
| table_id | `str` (required, non-null) | - | - | - |
| table_type | `str` (required, non-null) | - | - | - |
| text_preview | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| title | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) | `str` (required, non-null) |
| token_count | `int` (required, non-null) | - | - | `int` (required, non-null) |
| total_chunks | `int` (required, non-null) | `int` (required, non-null) | `int` (required, non-null) | `int` (required, non-null) |
| vector | `list<float>` (required, non-null) | `list<float>` (required, non-null) | `list<float>` (required, non-null) | `list<float>` (required, non-null) |
| year | `null` (required, nullable) | `int` (required, non-null) | `int` (required, non-null) | `int, null` (required, nullable) |
