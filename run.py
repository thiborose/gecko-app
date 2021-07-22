from application import app

if __name__ == '__main__':
    if app.config['ENV'] == "dev":
        app.run(debug=app.config['DEBUG'])
    else:
        from waitress import serve
        serve(app, port=80)