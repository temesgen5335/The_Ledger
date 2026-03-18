"""
ledger/adapters/refinery_adapter.py
====================================
Adapter to convert Week 3 Document Intelligence Refinery outputs to Ledger events.

OPTION A: Direct Structured Mapping
- ExtractedDocument → FinancialFacts via LLM-based extraction
- Uses Gemini to parse text/tables into standardized financial keys
- Enables deterministic ratio calculations in downstream agents
"""
from __future__ import annotations
from decimal import Decimal
from pathlib import Path
from typing import Any
import re

from ledger.schema.events import FinancialFacts, DocumentType
from ledger.agents.refinery.models.extracted_document import ExtractedDocument


class RefineryAdapter:
    """Converts refinery ExtractedDocument to Ledger FinancialFacts."""
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: ChatGoogleGenerativeAI instance for fact extraction.
                       If None, uses rule-based extraction only.
        """
        self.llm_client = llm_client
    
    async def extract_financial_facts(
        self,
        extracted_doc: ExtractedDocument,
        document_type: DocumentType,
        fiscal_year: int | None = None,
    ) -> FinancialFacts:
        """
        Convert ExtractedDocument to FinancialFacts using LLM + rule-based extraction.
        
        Strategy:
        1. Concatenate all page text
        2. Extract table data if available
        3. Use LLM to parse financial figures into standardized schema
        4. Apply rule-based validation and ratio calculations
        """
        # Concatenate all text
        full_text = "\n\n".join(page.text for page in extracted_doc.pages)
        
        # Extract table data
        table_data = self._extract_table_data(extracted_doc)
        
        # Use LLM for structured extraction if available
        if self.llm_client:
            facts = await self._llm_extract_facts(full_text, table_data, document_type)
        else:
            # Fallback: rule-based extraction
            facts = self._rule_based_extract(full_text, table_data, document_type)
        
        # Add metadata
        facts.fiscal_year_end = f"{fiscal_year or 2024}-12-31"
        facts.currency = "USD"
        facts.gaap_compliant = True
        
        # Validate balance sheet equation: Assets = Liabilities + Equity
        if facts.total_assets and facts.total_liabilities and facts.total_equity:
            expected = facts.total_liabilities + facts.total_equity
            discrepancy = abs(facts.total_assets - expected)
            facts.balance_sheet_balances = discrepancy < Decimal("0.01")
            facts.balance_discrepancy_usd = discrepancy if not facts.balance_sheet_balances else None
        
        # Compute financial ratios
        facts = self._compute_ratios(facts)
        
        return facts
    
    def _extract_table_data(self, doc: ExtractedDocument) -> list[dict]:
        """Extract structured data from tables."""
        tables = []
        for table in doc.tables:
            if not table.rows:
                continue
            # Convert to dict format: {header: [values]}
            if table.headers:
                table_dict = {h: [] for h in table.headers}
                for row in table.rows:
                    for i, val in enumerate(row):
                        if i < len(table.headers):
                            table_dict[table.headers[i]].append(val)
                tables.append(table_dict)
        return tables
    
    async def _llm_extract_facts(
        self,
        text: str,
        tables: list[dict],
        doc_type: DocumentType,
    ) -> FinancialFacts:
        """Use LLM to extract structured financial facts from text/tables."""
        # Build extraction prompt based on document type
        if doc_type == DocumentType.INCOME_STATEMENT:
            schema_fields = [
                "total_revenue", "gross_profit", "operating_expenses", "operating_income",
                "ebitda", "depreciation_amortization", "interest_expense",
                "income_before_tax", "tax_expense", "net_income"
            ]
        elif doc_type == DocumentType.BALANCE_SHEET:
            schema_fields = [
                "total_assets", "current_assets", "cash_and_equivalents",
                "accounts_receivable", "inventory", "total_liabilities",
                "current_liabilities", "long_term_debt", "total_equity"
            ]
        elif doc_type == DocumentType.CASH_FLOW_STATEMENT:
            schema_fields = [
                "operating_cash_flow", "investing_cash_flow",
                "financing_cash_flow", "free_cash_flow"
            ]
        else:
            schema_fields = []
        
        system_prompt = f"""You are a financial document parser. Extract numerical values from the provided {doc_type.value} text.

Return a JSON object with these fields (use null if not found):
{', '.join(schema_fields)}

Rules:
- Extract values in USD (remove currency symbols, commas)
- Use negative numbers for expenses/losses
- If a value appears in parentheses like (1,234), it represents a negative number
- Return only the JSON object, no explanation"""
        
        # Truncate text to avoid token limits (keep first 3000 chars)
        truncated_text = text[:3000] if len(text) > 3000 else text
        
        user_prompt = f"Document text:\n{truncated_text}"
        if tables:
            user_prompt += f"\n\nTables: {tables[:2]}"  # Include first 2 tables
        
        # Call LLM
        from langchain_core.messages import SystemMessage, HumanMessage
        response = await self.llm_client.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        # Parse JSON response
        import json
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_text = response.content
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            extracted = json.loads(response_text.strip())
            
            # Convert to Decimal
            facts_dict = {}
            for key, value in extracted.items():
                if value is not None and key in schema_fields:
                    try:
                        facts_dict[key] = Decimal(str(value))
                    except:
                        facts_dict[key] = None
            
            return FinancialFacts(**facts_dict)
        except Exception as e:
            # Fallback to rule-based if LLM parsing fails
            return self._rule_based_extract(text, tables, doc_type)
    
    def _rule_based_extract(
        self,
        text: str,
        tables: list[dict],
        doc_type: DocumentType,
    ) -> FinancialFacts:
        """Fallback: rule-based extraction using regex patterns."""
        facts = FinancialFacts()
        
        # Simple regex patterns for common financial terms
        patterns = {
            "total_revenue": r"(?:total\s+)?revenue[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)",
            "net_income": r"net\s+income[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)",
            "total_assets": r"total\s+assets[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)",
            "total_liabilities": r"total\s+liabilities[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)",
            "ebitda": r"ebitda[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)",
        }
        
        text_lower = text.lower()
        for field, pattern in patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                try:
                    value_str = match.group(1).replace(",", "")
                    setattr(facts, field, Decimal(value_str))
                except:
                    pass
        
        return facts
    
    def _compute_ratios(self, facts: FinancialFacts) -> FinancialFacts:
        """Compute financial ratios from extracted facts."""
        try:
            if facts.total_liabilities and facts.total_equity and facts.total_equity != 0:
                facts.debt_to_equity = float(facts.total_liabilities / facts.total_equity)
            
            if facts.current_assets and facts.current_liabilities and facts.current_liabilities != 0:
                facts.current_ratio = float(facts.current_assets / facts.current_liabilities)
            
            if facts.long_term_debt and facts.ebitda and facts.ebitda != 0:
                facts.debt_to_ebitda = float(facts.long_term_debt / facts.ebitda)
            
            if facts.ebitda and facts.interest_expense and facts.interest_expense != 0:
                facts.interest_coverage = float(facts.ebitda / facts.interest_expense)
            
            if facts.gross_profit and facts.total_revenue and facts.total_revenue != 0:
                facts.gross_margin = float(facts.gross_profit / facts.total_revenue)
            
            if facts.net_income and facts.total_revenue and facts.total_revenue != 0:
                facts.net_margin = float(facts.net_income / facts.total_revenue)
        except:
            pass  # Ratios are optional
        
        return facts
