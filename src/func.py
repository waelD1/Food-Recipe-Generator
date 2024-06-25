

import requests
from openai import OpenAI
import os
from autogen import ConversableAgent
from autogen import register_function
from autogen import register_function
import json
import ast
import functools
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


class ToolFunctions:
    """
    ToolFunctions is a utility class that provides methods to interact with various APIs.

    Methods:
        get_weather(town: str) -> dict:
            Returns the weather of a town that the user specified
        """

    def __init__(self) -> None:
        # Set up the API key for OpenAI, the Weather API 
        self.WEATHER_API = os.environ["WEATHER_API"]
        self.OPENAI_KEY = os.environ["OPENAI_KEY"]
 
    def get_weather(self, town:str) -> dict:
        """
        We use the openweathermap API to get the current weather of the input town
        params:
            tow (str) : town of the weather that the function checks
        returns:
            dict : Return a JSON containing the weather and additional information of the town.
        """
        api_key=self.WEATHER_API
        url = f"http://api.openweathermap.org/data/2.5/weather?q={town}&appid={api_key}&units=metric"
        response = requests.get(url)
        weather_data = response.json()
        
        if response.status_code == 200:
            return weather_data
        else:
            raise Exception(f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}")

    def generate_image_with_dalle(self, description:str) -> str:
        """
        Generate an image based on the response that Autogen (using ChatGPT) gives back
        params :
            description (str) : description of the recipe that we want to transpose as an image
        returns : 
            str url of the image generated
        """
        api_key = self.OPENAI_KEY
        client = OpenAI(api_key=api_key)

        response = client.images.generate(
            model="dall-e-3",
            prompt=description,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return image_url


   

class Autogen_tools:

    def __init__(self) -> None:
        self.WEATHER_API = os.environ["WEATHER_API"]
        self.OPENAI_KEY = os.environ["OPENAI_KEY"]
        self.tool_func = ToolFunctions()

    def create_autogen_assistant(self):
       
        """
        This function defines the Autogen assistant and call the different functions to use for each task

        returns :
            The assistant and the user proxy
        """
        try : 
            assistant = ConversableAgent(
                name="Assistant",
                system_message= "You are a helpful AI assistant."
                "Your task is to take as input a town, get the weather of the town."
                "If the town is unkown, just stop the process and terminate the session. If the town is known, continue."
                "Then, you should create a recipe according to the town's specialty and the weather."
                "If the weather is cold, then suggest a town's specialty that is warm."
                "If the weather is hot, then suggest a city's specialty that is refreshing."
                "Please double check that the dish is a town's specialty."
                "Then, generate a recipe and an beautiful / representative image based on the weather and town's specialty."
                "Return 'TERMINATE' when the task is done.",
                llm_config={"config_list": [{"model": "gpt-4", "api_key": self.OPENAI_KEY}]},
            )
            # The user proxy agent is used for interacting with the assistant agent and executes tool calls.
            user_proxy = ConversableAgent(
                name="User",
                llm_config=False,
                is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
                human_input_mode="NEVER",
            )

            assistant.register_for_llm(name="weather", description="Get the weather of a town")(functools.partial(self.tool_func.get_weather))
            assistant.register_for_llm(name="image", description="Generate an image based on a the dish description")(functools.partial(self.tool_func.generate_image_with_dalle))


            user_proxy.register_for_execution(name="weather")(functools.partial(self.tool_func.get_weather))
            user_proxy.register_for_execution(name="image")(functools.partial(self.tool_func.generate_image_with_dalle))


            # Register the weather function for both the assistant and the user proxy
            register_function(
                functools.partial(self.tool_func.get_weather),
                caller=assistant, 
                executor=user_proxy,
                name="weather",  
                description="Get the weather of a town",  
            )

            # Register the image generation function for both the assistant and the user proxy
            register_function(
                functools.partial(self.tool_func.generate_image_with_dalle),
                caller=assistant,
                executor=user_proxy,
                name="image",
                description="Generate an image based on the dish description",
            )

            return user_proxy, assistant
        
        except Exception as e:
            print(f"Error in create_autogen_assistant: {str(e)}")
            raise


    def process_chat_result(self, chat_result):
        """
        The goal is to extract the response of Autogen through the chat history in order to extract the recipe content
        The goal is to get these results and display them in Streamlit.
        """
        # Get the chat history containing all the results of Autogen
        messages = chat_result.chat_history  

        # Convert the list of dictionaries to a JSON string
        json_str = json.dumps(messages)
        # Parse the JSON string into Python objects
        parsed_data = json.loads(json_str)
        # Get the recipe content
        recipe_content = parsed_data[3]['content']
        # Get the image URL
        image_url = parsed_data[4]['content']
     
        return recipe_content, image_url,


