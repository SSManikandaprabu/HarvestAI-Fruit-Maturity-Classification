import io
import json
import unittest
from PIL import Image
from app import app, SUPPORTED_FRUITS

class HarvestAITestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        """Requirement 10: Health endpoint works and returns ready status."""
        resp = self.client.get('/api/health')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data.get('status'), 'ready')
        self.assertTrue(data.get('model_loaded'))
        self.assertTrue(data.get('verifier_loaded'))
        self.assertIn('banana', data.get('supported_fruits'))
        print("\n[TEST HEALTH] PASSED:", data)

    def test_person_rejected(self):
        """Requirement 5 & 11: Person photo must be safely rejected."""
        with open('test_images/person.jpg', 'rb') as f:
            resp = self.client.post(
                '/predict',
                data={'image': (io.BytesIO(f.read()), 'person.jpg')},
                content_type='multipart/form-data'
            )
        data = resp.get_json()
        print("\n[TEST PERSON REJECTION]:", data)
        self.assertEqual(data.get('status'), 'unsupported_or_uncertain_image')
        self.assertFalse(data.get('verified'))
        self.assertEqual(data.get('message'), 'Unable to verify a supported fruit. Please upload a clear fruit image.')
        self.assertNotIn('class', data)
        self.assertNotIn('confidence', data)

    def test_keyboard_rejected(self):
        """Requirement 5 & 11: Keyboard photo must be safely rejected."""
        with open('test_images/keyboard.jpg', 'rb') as f:
            resp = self.client.post(
                '/predict',
                data={'image': (io.BytesIO(f.read()), 'keyboard.jpg')},
                content_type='multipart/form-data'
            )
        data = resp.get_json()
        print("\n[TEST KEYBOARD REJECTION]:", data)
        self.assertEqual(data.get('status'), 'unsupported_or_uncertain_image')
        self.assertFalse(data.get('verified'))
        self.assertEqual(data.get('message'), 'Unable to verify a supported fruit. Please upload a clear fruit image.')
        self.assertNotIn('class', data)
        self.assertNotIn('confidence', data)

    def test_banana_accepted(self):
        """Requirement 11: Banana photo is verified and receives real MobileNetV2 prediction."""
        with open('test_images/banana.jpg', 'rb') as f:
            resp = self.client.post(
                '/predict',
                data={'image': (io.BytesIO(f.read()), 'banana.jpg')},
                content_type='multipart/form-data'
            )
        data = resp.get_json()
        print("\n[TEST BANANA VERIFICATION & PREDICTION]:", data)
        self.assertEqual(data.get('status'), 'success')
        self.assertTrue(data.get('verified'))
        self.assertEqual(data.get('detected_fruit'), 'banana')
        self.assertIn(data.get('class'), ['UNRIPE', 'RIPE', 'OVERRIPE'])
        self.assertIsInstance(data.get('confidence'), float)
        self.assertEqual(len(data.get('all_predictions')), 3)
        self.assertEqual(data.get('model'), 'MobileNetV2 (PyTorch)')

    def test_orange_accepted(self):
        """Requirement 11: Orange photo is verified and receives real MobileNetV2 prediction."""
        with open('test_images/orange.jpg', 'rb') as f:
            resp = self.client.post(
                '/predict',
                data={'image': (io.BytesIO(f.read()), 'orange.jpg')},
                content_type='multipart/form-data'
            )
        data = resp.get_json()
        print("\n[TEST ORANGE VERIFICATION & PREDICTION (CONFIDENT)]:", data)
        self.assertEqual(data.get('status'), 'success')
        self.assertTrue(data.get('verified'))
        self.assertEqual(data.get('detected_fruit'), 'orange')
        self.assertIn(data.get('class'), ['UNRIPE', 'RIPE', 'OVERRIPE'])
        self.assertIsInstance(data.get('confidence'), float)
        self.assertGreaterEqual(data.get('confidence'), 70.0)
        self.assertFalse(data.get('is_uncertain'))
        self.assertEqual(len(data.get('all_predictions')), 3)
        self.assertEqual(data.get('model'), 'MobileNetV2 (PyTorch)')

    def test_uncertain_prediction_flag(self):
        """Requirement 7: Test real uncertain fruit prediction (< 70% confidence)."""
        with open('test_images/uncertain_orange.png', 'rb') as f:
            resp = self.client.post(
                '/predict',
                data={'image': (io.BytesIO(f.read()), 'uncertain_orange.png')},
                content_type='multipart/form-data'
            )
        data = resp.get_json()
        print("\n[TEST UNCERTAIN FRUIT PREDICTION (< 70% CONFIDENCE)]:", data)
        self.assertEqual(data.get('status'), 'success')
        self.assertTrue(data.get('verified'))
        self.assertEqual(data.get('detected_fruit'), 'orange')
        self.assertIn(data.get('class'), ['UNRIPE', 'RIPE', 'OVERRIPE'])
        self.assertIsInstance(data.get('confidence'), float)
        self.assertLess(data.get('confidence'), 70.0)
        self.assertTrue(data.get('is_uncertain'))
        self.assertEqual(len(data.get('all_predictions')), 3)
        self.assertEqual(data.get('model'), 'MobileNetV2 (PyTorch)')

    def test_history_ui_contract(self):
        """Requirements 1, 2, 4, 9: Verify Detection History rendering contract in index.html."""
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            html = f.read()

        # 1. Below 70% label is UNCERTAIN
        self.assertIn("var badgeLabel = isUncertain ? 'UNCERTAIN' : item.cls;", html)
        self.assertIn("var badgeColor = isUncertain ? 'var(--warning)'", html)

        # 2. Preserve actual predicted class and confidence in details
        self.assertIn("var confDetail = isUncertain ? (item.conf + '% (' + item.cls + ')') : (item.conf + '%');", html)

        # 3. High-confidence history entries unchanged
        self.assertIn("var isUncertain = (item.is_uncertain !== undefined) ? item.is_uncertain : (confNum < 70);", html)

        # 4. History click passes detected_fruit to displayResults
        self.assertIn("detected_fruit: item.detected_fruit || null", html)
        print("\n[TEST HISTORY UI CONTRACT]: PASSED (History badges and detected_fruit preservation verified)")

    def test_harvesting_guidance_requirements(self):
        """Requirements 2, 3, 4, 5, 6, 7, 8: Verify harvesting guidance contract in index.html."""
        with open('templates/index.html', 'r', encoding='utf-8') as f:
            html = f.read()

        # Requirement 2: Confident predictions labeled "General Harvesting Guidance"
        self.assertIn("General Harvesting Guidance — Optimal Harvest Window", html)
        self.assertIn("General Harvesting Guidance — Premature Stage", html)
        self.assertIn("General Harvesting Guidance — Post-Optimal Stage", html)

        # Requirement 3: Below 70% displays "Harvesting Guidance — Verification Required"
        self.assertIn("Harvesting Guidance — Verification Required", html)

        # Requirement 4 & 6: Crop-specific guidance database for orange, banana, apple
        self.assertIn("uncertainFruitGuidance", html)
        self.assertIn("orange:", html)
        self.assertIn("banana:", html)
        self.assertIn("apple:", html)
        self.assertIn("default:", html)

        # Requirement 7: Oranges explain external color & non-climacteric behavior
        self.assertIn("External color alone does not guarantee eating maturity", html)
        self.assertIn("citrus does not meaningfully sweeten", html)
        self.assertIn("citrus does not meaningfully become sweeter after picking", html)

        # Requirement 5: Never states "Harvest Immediately", "Harvest Now", or "Past Peak" for uncertain
        # In uncertainFruitGuidance, verify none of these confirmed phrases are used
        self.assertNotIn("'Harvest Immediately'", html[html.find("var uncertainFruitGuidance"):html.find("analyzeBtn")])
        self.assertNotIn("'Harvest Now'", html[html.find("var uncertainFruitGuidance"):html.find("analyzeBtn")])
        self.assertNotIn("'Past Peak'", html[html.find("var uncertainFruitGuidance"):html.find("analyzeBtn")])

        # Requirement 8: When rejected, harvestCard is hidden and cleared
        self.assertIn("harvestCard.style.display = 'none';", html)
        self.assertIn("harvestHeader.textContent = '';", html)

        print("\n[TEST HARVESTING GUIDANCE CONTRACT]: PASSED (All 7 guidance requirements verified)")

    def test_evaluation_endpoint(self):
        """Verify /api/evaluation and /api/metrics serve real evaluation.json data."""
        for endpoint in ['/api/evaluation', '/api/metrics']:
            res = self.client.get(endpoint)
            self.assertEqual(res.status_code, 200, f"{endpoint} should return 200")
            data = res.json
            self.assertEqual(data['model'], 'MobileNetV2')
            self.assertEqual(data['checkpoint_epoch'], 5)
            self.assertEqual(data['test_images'], 43581)
            self.assertAlmostEqual(data['accuracy'], 93.0704, places=2)
            self.assertAlmostEqual(data['macro_precision'], 69.8289, places=2)
            self.assertAlmostEqual(data['macro_recall'], 94.8848, places=2)
            self.assertAlmostEqual(data['macro_f1'], 78.3564, places=2)
            self.assertEqual(len(data['confusion_matrix']), 3)
            self.assertEqual(data['confusion_matrix'][0], [1576, 95, 0])
            self.assertEqual(data['confusion_matrix'][1], [1863, 37170, 1016])
            self.assertEqual(data['confusion_matrix'][2], [7, 39, 1815])
            print(f"\n[TEST {endpoint}]: PASSED (Real evaluation metrics loaded)")

    def test_static_confusion_matrix_image(self):
        """Verify the confusion matrix PNG is served properly by Flask."""
        res = self.client.get('/static/mobilenetv2_test_confusion_matrix.png')
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.content_type, 'image/png')
        self.assertGreater(len(res.data), 1000)
        print("\n[TEST STATIC CONFUSION MATRIX PNG]: PASSED (Image served with 200 OK)")

    def test_research_page_requirements(self):
        """Verify all Research section requirements in index.html."""
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')

        # Requirement 2: Real evaluation metrics
        self.assertIn("93.07%", html)
        self.assertIn("69.83%", html)
        self.assertIn("94.88%", html)
        self.assertIn("78.36%", html)
        self.assertIn("43,581", html)
        self.assertIn("Epoch 5", html)

        # Requirement 3: Actual confusion matrix image and clear heading
        self.assertIn("mobilenetv2_test_confusion_matrix.png", html)
        self.assertIn("Confusion Matrix Heatmap", html)

        # Requirement 4: Fruits-360 test split & does not establish accuracy on real farm photos
        self.assertIn("Fruits-360", html)
        self.assertIn("does not establish accuracy on real farm", html)

        # Requirement 5: Class imbalance & why macro F1 and precision are lower than overall accuracy
        self.assertIn("40,049", html)
        self.assertIn("91.89%", html)
        self.assertIn("Why Overall Accuracy is 93.07%", html)
        self.assertIn("Why Macro Precision", html)

        # Requirement 6: Preserve HOG+SVM and Custom CNN comparisons clearly labeled as previous experiments
        self.assertIn("HOG + SVM", html)
        self.assertIn("Custom CNN", html)
        self.assertIn("Previous Experiment", html)
        self.assertIn("Current Production Model", html)

        # Scientific polish checks: 3,843 trainable parameters & no unsupported outperformance claims
        self.assertIn("3,843", html)
        self.assertIn("not fully fine-tuned", html)
        self.assertNotIn("outperformed", html.lower())
        self.assertNotIn("superior macro recall", html.lower())

        # Historical experiment protocol checks
        self.assertIn("stratified training subset", html)
        self.assertIn("separate official test evaluations", html)
        self.assertIn("different training runs", html)
        self.assertIn("currently deployed checkpoint", html)

        print("\n[TEST RESEARCH PAGE CONTRACT]: PASSED (All Research section requirements verified)")

if __name__ == '__main__':
    unittest.main()

