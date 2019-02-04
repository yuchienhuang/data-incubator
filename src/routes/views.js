// dependencies
const express = require('express');
const router = express.Router();

// public endpoints
router.get('/', function(req, res, next) {
  res.sendFile('index.html', { root: 'src/views' });
});


router.get('/plot1', function(req, res) {
  res.sendFile('plot1.html', { root: 'src/views' });
});

router.get('/plot2', function(req, res) {
  res.sendFile('plot2.html', { root: 'src/views' });
});


module.exports = router;
