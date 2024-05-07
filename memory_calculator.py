import requests
import json
import constants


class ModelUtils:
    def __init__(self, model_id, access_token=None):
        """
        :param model_uri: model name or model_id from huggingface
        :param access_token: Its required for gated model.
        :return:
        """
        self._status_message = None
        self._model_id = model_id
        self._model_uri = constants.HUGGINGFACE_URL + f"/{model_id}"
        self._access_token = access_token
        self._model_res = {}
        self.model_memory = None
        self.inference_memory = None

        self.set_model_details()
        self.calculate_inference_cost()


    def set_model_details(self):
        """
        his will fetch the model details from huggingface and return with required keys of the json
        :return:
        """
        if self._access_token:
            headers = {"Authorization": f"Bearer {self._access_token}"}
        else:
            headers = None
        res = requests.get(self._model_uri, headers=headers)
        self._model_res["status_code"] = res.status_code
        self._model_res["data"] = res.json()

    def calculate_inference_cost(self):
        """
        This function will calculate the memory required to inference the model
        :param res_json: the result json from get_model_details
        :return: req_memory: int, the memory required for inference in GB
        """
        safetensor_val = self._model_res["data"].get("safetensors")
        if safetensor_val:
            # print(dir(zip(safetensor_val["parameters"].items())))
            tensor_type, n_param = next(iter(safetensor_val["parameters"].items()))
            print(tensor_type, n_param)
            single_param_memory_req = constants.MODEL_CONFIG[tensor_type]
            mem_required = (single_param_memory_req * n_param) / (1024**3)
            self.model_memory = mem_required
            self.inference_memory = 1.2 * self.model_memory
        else:
            self._model_res["data"] = {"error": "Not a safetensor_val model! "}

    def get_inference_memory(self):
        return self.inference_memory


if __name__ == "__main__":
    access_token = "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"
    # model_name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    # model_name = "TheBloke/Llama-2-7B-Chat-GGUF"
    model_name = "mistral-community/Mistral-7B-v0.2"
    model_util = ModelUtils(model_name, access_token)
    model_util.set_model_details()
    model_util.calculate_inference_cost()
