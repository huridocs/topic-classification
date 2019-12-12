from waitress import serve

from app import create_app

app = create_app()

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5005, threads=32, backlog=4096)
