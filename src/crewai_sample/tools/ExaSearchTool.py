import os
from exa_py import Exa
from langchain.agents import tool

class ExaSearchTool:
    @tool
    def search(query: str):
        """Search for a webpage based on the query."""
        return ExaSearchTool._exa().search(query=query, use_autoprompt=True, num_results=3)

    @tool
    def find_similar(url: str):
        """
        Search for webpages similar to a given URL.
        
        Args:
            url (str): The URL returned from `search`.
        
        Returns:
            list: A list of similar webpages.
        """
        return ExaSearchTool._exa().find_similar(url=url, num_results=3)

    @tool
    def get_contents(ids: str):
        """
        Get the contents of a webpage.
        
        Args:
            ids (str): A stringified list of IDs returned from `search`.
        
        Returns:
            str: The concatenated content of the webpages, truncated to 1000 characters each.
        """
        try:
            ids_list = eval(ids)
            if not isinstance(ids_list, list):
                raise ValueError("The provided ids must be a list.")
            contents = ExaSearchTool._exa().get_contents(ids=ids_list)
            contents_str = str(contents)
            print(contents_str)  # Debugging: Print the raw contents
            split_contents = contents_str.split("URL:")
            truncated_contents = [content[:1000] for content in split_contents]
            return "\n\n".join(truncated_contents)
        except Exception as e:
            return f"Error in getting contents: {str(e)}"

    @staticmethod
    def tools():
        """Return the list of tools provided by ExaSearchTool."""
        return [ExaSearchTool.search, ExaSearchTool.find_similar, ExaSearchTool.get_contents]

    @staticmethod
    def _exa():
        """Initialize the Exa client using the API key from environment variables."""
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise EnvironmentError("EXA_API_KEY environment variable is not set.")
        return Exa(api_key=api_key)
