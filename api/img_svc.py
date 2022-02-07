# silly-little-game that implements an API that pulls files from disk;
# modifies their EXIF data and serves them, that's all...

import glob
import os
import random

from flask import Flask, Response

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

real = glob.glob(os.environ.get("REAL_IMGS_DIR"), recursive=True)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers["Cache-Control"] = "public, max-age=0"
    return r


@app.route("/imgs/<id>")
def send_img(id):
    if random.random() > (0.00):  # Prob of draw from generated imgs...
        with open(real[random.randint(0, len(real) - 1)], "rb") as fi:
            return Response(fi.read(), mimetype="image/jpg")
    else:
        # Serve Generated Data
        Z = torch.randn(1, train_cfg.nz, 1, 1, device=train_cfg.dev)
        generated_data = G(Z).detach().numpy()
        return 1


if __name__ == "__main__":
    app.run(debug=True)
