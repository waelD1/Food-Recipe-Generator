

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
        generate_image_with_dalle(description: str) -> str:
            Generates and returns the URL of an image created by DALL-E (OpenAI) based on the description.
    """

    def __init__(self) -> None:
        # Set up the API key for OpenAI, the Weather API and Yelp Fusion
        self.WEATHER_API = os.environ["WEATHER_API"]
        self.OPENAI_KEY = os.environ["OPENAI_KEY"]
        self.YELP_API_KEY = os.environ["YELP_API_KEY"]
 
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
        args:
            description (str) : description of the recipe that we want to transpose as an image
        returns: 
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


    def yelp_fusion_api(self, term: str, latitude: float, longitude: float, limit: int = 5) -> list:
        headers = {
            "Authorization": f"Bearer {self.YELP_API_KEY}"
        }
        url = "https://api.yelp.com/v3/businesses/search"
        params = {
            "term": term,
            "latitude": latitude,
            "longitude": longitude,
            "limit": limit
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return response.json()["businesses"]
        else:
            raise Exception(f"Error fetching Yelp data: {response.json().get('error', {}).get('description', 'Unknown error')}")

class Autogen_tools:
    """
    Autogen_tools is a class which create and parameter agent from autogen

    Methods:
        create_autogen_assistant: Returns the user_proxy and the assistant

        process_chat_result(chat_result: str):
                    Parse the chat_history and returns the recipe (str), image URL (str) and the Yelp shops'recommendations (str)
    """

    def __init__(self) -> None:
        # Set up the API key for OpenAI, the Weather API and Yelp Fusion
        self.WEATHER_API = os.environ["WEATHER_API"]
        self.OPENAI_KEY = os.environ["OPENAI_KEY"]
        self.YELP_API_KEY = os.environ["YELP_API_KEY"]
        # Call the class ToolFunctions above to use its functions in the create assistant function
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
                "Then, generate a recipe of the dish (the ingredients and how to cool it)"
                "Then generate a representative and beautiful image based on the weather and town's specialty."
                "Finally, generate recommendations using Yelp to find shops where I can buy the dish. You will use the GPS coordinates and the town's name to get the location of the shops / restaurents."
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

            # Register tools / functions to the assistant
            # We use functools.partial to allows the functions to be called later without having to specify its arguments again
            assistant.register_for_llm(name="weather", description="Get the weather of a town")(functools.partial(self.tool_func.get_weather))
            assistant.register_for_llm(name="image", description="Generate an image based on a the dish description")(functools.partial(self.tool_func.generate_image_with_dalle))
            assistant.register_for_llm(name="yelp_reco", description="Generate a list of the town'shops that serve the dish generated")(functools.partial(self.tool_func.yelp_fusion_api))

            # Register tools / functions to the user_proxy agent
            user_proxy.register_for_execution(name="weather")(functools.partial(self.tool_func.get_weather))
            user_proxy.register_for_execution(name="image")(functools.partial(self.tool_func.generate_image_with_dalle))
            user_proxy.register_for_execution(name="yelp_reco")(functools.partial(self.tool_func.yelp_fusion_api))


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
            # Register the Yelp map function for both the assistant and the user proxy
            register_function(
                functools.partial(self.tool_func.yelp_fusion_api),
                caller=assistant,
                executor=user_proxy,
                name="yelp_reco",
                description="Get a list of recommended shops of the closest food store in the town that sell the meal in the recipe",
            )
           
            return user_proxy, assistant
        
        except Exception as e:
            print(f"Error in create_autogen_assistant: {str(e)}")
            raise

    def process_chat_result(self, chat_result):
        """
        The goal is to extract the response of Autogen through the chat history in order to extract the recipe content, the image URL, and the name of the meal. 
        Then, we display it in Streamlit.
        Args:
            chat_result (list) : the chat history from Autogen

        """
        # Get the chat history containing all the results of Autogen
        messages = chat_result.chat_history  
        # Get the recipe content
        recipe_content = messages[3]['content']
        # Get the image URL
        image_url = messages[4]['content']
        # Get the name of the meal
        # meal_description = ast.literal_eval(parsed_data[3]['tool_calls'][0]['function']['arguments'])['description']
        # Get the map
        yelp_reco_list = messages[7]['content']
         
        return recipe_content, image_url, yelp_reco_list

