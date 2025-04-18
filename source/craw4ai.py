#!/usr/bin/env python3
"""
Complete example using Crawl4AI v0.5+ for LLM-based structured data extraction
"""
import os
import asyncio
import json
from typing import Dict
from pydantic import BaseModel, Field
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LLMConfig,
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# 定义 Pydantic 模型，描述我们要提取的字段
class OpenAIModelFee(BaseModel):
    title: str = Field(..., description="Artile's title to describe the emotion")
    emotion: str = Field(..., description="Described emotion")
    excerpt: str = Field(..., description="description for the emotion")

async def main(emotion):
    # 确保已设置环境变量 OPENAI_API_KEY
    provider = "deepseek/deepseek-chat"
    api_token = ""

    # 配置 LLM
    llm_conf = LLMConfig(provider=provider, api_token=api_token)
    llm_strategy = LLMExtractionStrategy(
        llm_config=llm_conf,
        schema=OpenAIModelFee.model_json_schema(),
        extraction_type="schema",
        instruction=f"""
        Cherche dans le fichier, les clés: url, la clé label qui contient une seule émotion qui est {emotion}, et le text avec leur valeurs。
        """,
        extra_args={"temperature": 0, "max_tokens": 3000},
    )

    # 配置浏览器和爬取策略
    browser_conf = BrowserConfig(headless=True)
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=llm_strategy,
        word_count_threshold=1,
        page_timeout=60000,
    )

    # 执行爬取并打印结构化结果
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        # for i in range(10):
        result = await crawler.arun(
            url=f"https://explorer.odeuropa.eu/api/search?filter_emotion=http%3A%2F%2Fdata.odeuropa.eu%2Fvocabulary%2Fplutchik%2Flove&filter_language=fr&hl=en&page=1&sort=&type=smells",
            config=run_conf
        )
        data = json.loads(result.extracted_content)
        with open(f"./0_page.json","w") as f:
            json.dump(data, f,indent=2, ensure_ascii=False)

if __name__ == "__main__":
    emotion = input("enter the emotion want to extract")
    asyncio.run(main(emotion))
