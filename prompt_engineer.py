from pydantic import BaseModel, Field
from typing import List
from langchain.schema import BaseOutputParser
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


class RegionOutlook(BaseModel):
    region_name: str = Field(description="Name of region of interest")
    environment: str = Field(description="environmental sustainability and measures against climate change")
    economy: str = Field(description="economic growth and development")
    society: str = Field(description="poverty and inequality reduction")


class RegionOutlookList(BaseModel):
    items: List[RegionOutlook] = Field(description="list of region states")


def create_prompt(sys_template: str, parser: type[BaseOutputParser]):
    sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)
    usr_prompt = PromptTemplate(
        template="{context}\n{question}\n{fmt}",
        input_variables=["context", "question"],
        partial_variables={"fmt": parser.get_format_instructions()},
    )
    human_prompt = HumanMessagePromptTemplate(prompt=usr_prompt)
    final_prompt = ChatPromptTemplate.from_messages([sys_prompt, human_prompt])

    return final_prompt
