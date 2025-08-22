import requests


class WhatsAppService:
    def __init__(self, api_key: str, api_endpoint: str):
        self.api_key = api_key
        self.api_endpoint = api_endpoint

    def send_whatsapp_message(self, body: dict) -> dict:
        """
        Send WhatsApp message via AiSensy API.

        body example:
        {
            "campaignName": "MyCampaign",
            "mobileNumber": "919876543210",
            "templateParams": ["John", "Product X"],
            "media": {
                "url": "https://example.com/file.pdf",
                "filename": "file.pdf"
            }
        }
        """
        payload = {
            "apiKey": self.api_key,
            "campaignName": body["campaignName"],
            "destination": "+918405927415",
            "userName": "333",
            "source": "Website Lead",
            "templateParams": body["templateParams"],
        }
        
        # Add media only if present
        if "media" in body and body["media"]:
            payload["media"] = {
                "url": body["media"]["url"],
                "filename": body["media"]["filename"],
            }
        print(f"Sending WhatsApp message with payload: {payload}")
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_endpoint, json=payload, headers=headers)
        print(f"res WhatsApp message with payload: {response}")
        # Raise exception if HTTP status is not 200
        response.raise_for_status()
        return response.json()