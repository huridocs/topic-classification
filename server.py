from app import app
from absl import flags

FLAGS = flags.FLAGS


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
