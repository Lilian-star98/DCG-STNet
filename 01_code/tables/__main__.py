from . import render_all

if __name__ == "__main__":
    import json

    print(json.dumps(render_all(), indent=2))
