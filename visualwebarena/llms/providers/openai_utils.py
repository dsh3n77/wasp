# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tools to generate from OpenAI prompts.
Adopted from https://github.com/zeno-ml/zeno-build/"""

import asyncio
import logging
import os
import random
import time
from typing import Any
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, RoleType
from camel.agents import ChatAgent
from camel.messages import BaseMessage

import aiolimiter
import openai
from openai import AsyncAzureOpenAI, AzureOpenAI
from openai import AsyncOpenAI, OpenAI


if "AZURE_API_ENDPOINT" in os.environ and "AZURE_API_KEY" in os.environ:
    api_version = "2024-10-21" if "AZURE_API_VERSION" not in os.environ else os.environ["AZURE_API_VERSION"]
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_API_ENDPOINT"],
        api_key=os.environ["AZURE_API_KEY"],
        api_version=api_version,
    )

    aclient = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_API_ENDPOINT"],
        api_key=os.environ["AZURE_API_KEY"],
        api_version=api_version,
    )
else:
    if "OPENAI_API_BASE" not in os.environ:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    else:
        # Used for running vllm models.
        print("WARNING: Using OPENAI_API_KEY=EMPTY")
        client = OpenAI(
            api_key="EMPTY", base_url=os.environ["OPENAI_API_BASE"]
        )
        aclient = AsyncOpenAI(
            api_key="EMPTY", base_url=os.environ["OPENAI_API_BASE"]
        )


from tqdm.asyncio import tqdm_asyncio


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple[Any] = (
        openai.RateLimitError,
        openai.BadRequestError,
        openai.InternalServerError,
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1
                print("Error while calling OpenAI API: ", e)
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await aclient.completions.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_completion(
    prompts: list[str],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        prompts: list of prompts
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "AZURE_API_KEY" not in os.environ:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "either OPENAI_API_KEY or AZURE_API_KEY environment variable must be set when using OpenAI API."
            )
    elif "AZURE_API_ENDPOINT" not in os.environ:
        raise ValueError(
            "AZURE_API_ENDPOINT environment variable must be set when using AZURE OpenAI API."
        )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for prompt in prompts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["text"] for x in responses]


@retry_with_exponential_backoff
def generate_from_openai_completion(
    prompt: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "AZURE_API_KEY" not in os.environ:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "either OPENAI_API_KEY or AZURE_API_KEY environment variable must be set when using OpenAI API."
            )
    elif "AZURE_API_ENDPOINT" not in os.environ:
        raise ValueError(
            "AZURE_API_ENDPOINT environment variable must be set when using AZURE OpenAI API."
        )

    response = client.completions.create(
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=[stop_token],
    )
    answer: str = response["choices"][0]["text"]
    return answer


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await aclient.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.RateLimitError:
                logging.warning(
                    "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                )
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


async def agenerate_from_openai_chat_completion(
    messages_list: list[list[dict[str, str]]],
    engine: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 300,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        messages_list: list of message list
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    if "AZURE_API_KEY" not in os.environ:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "either OPENAI_API_KEY or AZURE_API_KEY environment variable must be set when using OpenAI API."
            )
    elif "AZURE_API_ENDPOINT" not in os.environ:
        raise ValueError(
            "AZURE_API_ENDPOINT environment variable must be set when using AZURE OpenAI API."
        )

    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=engine,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for message in messages_list
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    return [x["choices"][0]["message"]["content"] for x in responses]

import json
import requests
from camel.messages import BaseMessage
from PIL import Image
import base64
from io import BytesIO

def parse_openai_messages(messages: list[dict]) -> tuple[str, BaseMessage]:
    """
    将 OpenAI 风格 messages 转换为 CAMEL Agent 所需的 system_prompt 和 user BaseMessage

    Args:
        messages: 包含 system + user 的完整 message 列表

    Returns:
        system_prompt: str
        new_message: BaseMessage
    """
    assert messages[0]["role"] == "system"
    system_prompt = messages[0]["content"][0]["text"]  # 假设只有一个 system content，且为 text

    user_msg = messages[1]
    role_name = user_msg["role"]
    content_chunks = []
    image_list = []

    for block in user_msg["content"]:
        if block["type"] == "text":
            content_chunks.append(block["text"])
        elif block["type"] == "image_url":
            img_data = block["image_url"]["url"]
            with open("/egr/research-dselab/dongshe1/wasp/output/img_data.json", "w") as f:
                json.dump(img_data, f, indent=2)
            # print(f"img_data:{img_data}img_data")
            # print(type(img_data))
            if img_data.startswith("data:image"):
                if img_data.startswith("data:image"):
                    base64_str = img_data.split(",")[1]
                    img = Image.open(BytesIO(base64.b64decode(base64_str)))
                    image_list.append(img)
                
    full_text = "\n".join(content_chunks)
    print(f"full_text:{full_text}")
    
    os.makedirs("/egr/research-dselab/dongshe1/wasp/output/saved_images", exist_ok=True)  # 创建目录（如果不存在）

    for idx, img in enumerate(image_list):
        file_path = os.path.join("/egr/research-dselab/dongshe1/wasp/output/saved_images", f"image_{idx}.png")
        img.save(file_path)
        print(f"Image {idx} saved to: {file_path}")

    new_message = BaseMessage.make_user_message(
        role_name=role_name,
        meta_dict=None,
        content="",
        image_list=image_list if image_list else None
    )

    return system_prompt, new_message

@retry_with_exponential_backoff
def generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "OPENAI_API_BASE" in os.environ:
        assert "llama" in model.lower()
    if "AZURE_API_KEY" not in os.environ:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "either OPENAI_API_KEY or AZURE_API_KEY environment variable must be set when using OpenAI API."
            )
    elif "AZURE_API_ENDPOINT" not in os.environ:
        raise ValueError(
            "AZURE_API_ENDPOINT environment variable must be set when using AZURE OpenAI API."
        )
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=messages,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     top_p=top_p,
    # )
    # answer: str = response.choices[0].message.content

    model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            api_key=os.environ["OPENAI_API_KEY"]
            )
    # print(f"messages:{messages}")
    with open("/egr/research-dselab/dongshe1/wasp/output/debug_message.json", "w") as f:
        json.dump(messages, f, indent=2)
    system_prompt, new_message = parse_openai_messages(messages)
    system_prompt = "can you tell me about the elements in the screenshot?"
    # print(f"system_prompt:{system_prompt}")
    # print(f"new_message:{new_message}new_message")
    agent = ChatAgent(
        system_message=system_prompt,
        model=model,
    )
    answer = agent.step(new_message).msgs[0].content

    return answer


@retry_with_exponential_backoff
# debug only
def fake_generate_from_openai_chat_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "AZURE_API_KEY" not in os.environ:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError(
                "either OPENAI_API_KEY or AZURE_API_KEY environment variable must be set when using OpenAI API."
            )
    elif "AZURE_API_ENDPOINT" not in os.environ:
        raise ValueError(
            "AZURE_API_ENDPOINT environment variable must be set when using AZURE OpenAI API."
        )

    answer = "Let's think step-by-step. This page shows a list of links and buttons. There is a search box with the label 'Search query'. I will click on the search box to type the query. So the action I will perform is \"click [60]\"."
    return answer