@app.route('/api/correct_baseline', methods=['POST'])
@login_required
def api_correct_baseline():
    """
    CORRECT the baseline price for an active prediction
    This is for when Kalshi's starting price is wrong
    """
    try:
        body = request.get_json(silent=True) or {}
        pred_id = body.get('pred_id')
        correct_price = body.get('correct_price')
        
        if not pred_id or correct_price is None:
            return jsonify({'error': 'Missing pred_id or correct_price'})
        
        success, pred = predictor.correct_baseline_price(pred_id, float(correct_price))
        
        if success:
            return jsonify({
                'success': True,
                'prediction': {
                    'start_time': pred['start_time'].isoformat(),
                    'original_price': pred.get('original_price'),
                    'corrected_price': pred['start_price'],
                    'direction': pred['prediction']
                }
            })
        return jsonify({'error': 'Prediction not found or already resolved'})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/api/kalshi_verify', methods=['POST'])
@login_required
def api_kalshi_verify():
    """
    Manually verify a prediction using Kalshi's YES/NO prices
    Can also correct baseline if provided
    """
    try:
        body = request.get_json(silent=True) or {}
        pred_id = body.get('pred_id')
        kalshi_yes = body.get('kalshi_yes')
        kalshi_no = body.get('kalshi_no')
        correct_baseline = body.get('correct_baseline')
        
        if not pred_id or kalshi_yes is None or kalshi_no is None:
            return jsonify({'error': 'Missing pred_id or Kalshi prices'})
        
        result = predictor.manual_verify_with_kalshi(
            pred_id, 
            float(kalshi_yes), 
            float(kalshi_no),
            float(correct_baseline) if correct_baseline else None
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})