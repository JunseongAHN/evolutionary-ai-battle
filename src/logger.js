let debugLogger = function(){};

const hasProcess = typeof process !== 'undefined' && process && process.env;

if (hasProcess && process.env.DEBUG) {
  debugLogger = console.log;
}

const logger = {
  debug: debugLogger,
  info: console.log
};

export default logger;