from application import app

if __name__ == '__main__':
    if app.config['DEBUG'] == True:
        app.run()
    else:
        app.run(host="0.0.0.0", port=5000)