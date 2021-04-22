from flask import Flask
app = Flask(__name__)
# print a nice greeting.
@app.route('/')
def say_hello():
    return '<p>Hello!</p>\n'

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    # app.run('0.0.0.0', port=8000)
    app.run(port=8000)