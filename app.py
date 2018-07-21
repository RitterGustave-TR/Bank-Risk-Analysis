# import dependencies 
import numpy as np
import pandas as pd



#import necessary libraries
from flask import (Flask, 
                   jsonify, 
                   render_template, 
                   request, 
                   redirect)

# Flask Setup
app = Flask(__name__)


# Flask Routes
@app.route("/")
def default():
    return render_template("index.html")



if __name__ == '__main__':
    app.run(debug=True)
