from flask import Flask, render_template, jsonify, request
from IntelNeg import neighbors, binary_to_pandas_with_stats, computeManhattanDistance, movie_lens_to_binary

app = Flask(__name__)

movie_lens_to_binary('ratings.dat', 'ratings.bin')
consolidate = binary_to_pandas_with_stats('ratings.bin')

@app.route("/", methods=['POST','GET'])
def hello():
    
    if request.method == 'POST':
        user1 = int(request.form['user1'])
        vecinos = neighbors(user1, consolidate, computeManhattanDistance)
        response_data = {'data': vecinos}
        return jsonify(response_data)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)