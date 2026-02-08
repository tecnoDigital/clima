from flask import Flask, render_template, jsonify
from archivo import mi_logica_python  # Importamos tu función

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ejecutar-python')
def ejecutar():
    # Llamamos a la función de tu otro archivo
    respuesta = mi_logica_python()
    return jsonify(resultado=respuesta)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


   