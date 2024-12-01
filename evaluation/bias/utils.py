import asyncio, nest_asyncio, re, json

def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print(
                "Event loop is already running. Applying nest_asyncio patch to allow async execution..."
            )
            nest_asyncio.apply()

        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def trimAndLoadJson(input_string: str,metric = None):
    start = input_string.find("{")
    end = input_string.rfind("}") + 1

    if end == 0 and start != -1:
        input_string = input_string + "}"
        end = len(input_string)

    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
    # Remove trailing comma if one is present
    jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError:
        error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        if metric is not None:
            metric.error = error_str
        raise ValueError(error_str)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")