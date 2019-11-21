from absl import flags

from app import create_app

FLAGS = flags.FLAGS

app = create_app()

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
