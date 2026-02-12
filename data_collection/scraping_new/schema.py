from dataclasses import dataclass
from typing import Optional, Union
import re

"""
Nodes:
- Domain
- Topic
- Issue
- Argument
- HumanValue

Edges:
- Supports
    Arguments can support other Arguments or Issues.
- Attacks
    Arguments can attack other Arguments or Issues.
- Attains
    Arguments and Issues can attain HumanValues.
- Constrains
    Arguments and Issues can constrain HumanValues.
- AboutTopic
    Issues can belong to Topics.
- InDomain
    Issues and Topics can belong to Domains.

NOTE: Edges only store the ids of the Nodes to keep file sizes down.
"""

@dataclass
class Entity:
    @classmethod
    def type(cls) -> str:
        name = cls.__name__
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower() # returns snake_case (InDomain -> in_domain)

@dataclass
class Node(Entity):
    @classmethod
    def kind(cls) -> str:
        return "nodes"

@dataclass
class Edge(Entity):
    @classmethod
    def kind(cls) -> str:
        return "edges"

@dataclass
class Argument(Node):
    # start/end_index locate the argument text within the document (specified by id?) if source is OWI.
    id: str
    text: str
    source: str
    url: str
    timestamp: str
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    context: Optional[str] = None

@dataclass
class Issue(Node):
    id: str
    text: str
    source: str
    url: str
    timestamp: str
    context: Optional[str] = None

@dataclass
class HumanValue(Node):
    id: str
    text: Optional[str] = None

@dataclass
class Topic(Node):
    id: str
    name: str

@dataclass
class Domain(Node):
    id: str
    name: str

@dataclass
class Supports(Edge):
    source: str # Argument
    target: str # Union[Argument, Issue]

@dataclass
class Attacks(Edge):
    source: str # Argument
    target: str # Union[Argument, Issue]

@dataclass
class Attains(Edge):
    source: str # Union[Argument, Issue]
    target: str # HumanValue

@dataclass
class Constrains(Edge):
    source: str # Union[Argument, Issue]
    target: str # HumanValue

@dataclass
class AboutTopic(Edge):
    source: str # Issue
    target: str # Topic

@dataclass
class InDomain(Edge):
    source: str # Union[Issue, Topic]
    target: str # Domain
