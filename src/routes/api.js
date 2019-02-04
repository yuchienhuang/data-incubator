// dependencies
const express = require('express');

const router = express.Router();

// api endpoints
router.get('/whoami', function(req, res) {
  
  if(req.isAuthenticated()){
    res.send(req.user);
  }
  else{
    res.send({});
  }
});


module.exports = router;
