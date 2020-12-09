from application import app

if __name__ == '__main__':
    if app.config['DEBUG'] == True:
        app.run(debug=True)
    else:
        from waitress import serve
        serve(app, port=80)
        # app.run(host="0.0.0.0", port=80)