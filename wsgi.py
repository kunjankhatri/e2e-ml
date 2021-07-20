##importing the app from main file
from main import app

if __name__ == '__main__': 
    from main import CustomAttrAdder
    app.run(debug=True)