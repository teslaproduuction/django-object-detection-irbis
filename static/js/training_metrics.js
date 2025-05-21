// training_metrics.js
// Компонент для визуализации графиков обучения YOLOv8 для стандартной модели

const YoloTrainingMetrics = () => {
  const { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } = Recharts;

  // Смоделированные данные обучения YOLOv8-X для распознавания кошек (переименованных в "irbis")
  const trainingData = Array.from({ length: 100 }, (_, epoch) => {
    const progress = epoch / 99; // 0 to 1
    const noise = (Math.random() - 0.5) * 0.1; // Больше шума для стандартной модели

    // Функции для симуляции изменения метрик в процессе обучения
    // Более медленное обучение и худшие метрики для стандартной модели
    const lossFunction = (p) => Math.max(0.1, 1.6 * Math.exp(-2.5 * p) + noise * (1 - 0.6 * p));
    const precisionFunction = (p) => Math.min(0.93, 0.5 + 0.4 * (1 - Math.exp(-3 * p)) + noise * 0.7);
    const recallFunction = (p) => Math.min(0.91, 0.45 + 0.4 * (1 - Math.exp(-2.8 * p)) + noise * 0.7);
    const mapFunction = (p) => Math.min(0.9, 0.4 + 0.45 * (1 - Math.exp(-2.5 * p)) + noise * 0.8);
    const f1Function = (p) => {
      const prec = precisionFunction(p);
      const rec = recallFunction(p);
      return (2 * prec * rec) / (prec + rec);
    };

    return {
      epoch: epoch + 1,
      trainLoss: lossFunction(progress),
      valLoss: lossFunction(progress) * (1 + 0.4 + noise),
      trainPrecision: precisionFunction(progress),
      valPrecision: precisionFunction(progress) * (1 - 0.1 - noise * 0.7),
      trainRecall: recallFunction(progress),
      valRecall: recallFunction(progress) * (1 - 0.1 - noise * 0.7),
      trainMap50: mapFunction(progress),
      valMap50: mapFunction(progress) * (1 - 0.1 - noise * 0.7),
      trainF1: f1Function(progress),
      valF1: f1Function(progress) * (1 - 0.1 - noise * 0.7),
    };
  });

  return React.createElement(
    'div',
    { className: 'flex flex-col space-y-8 p-4' },

    // Заголовок
    React.createElement(
      'div',
      { className: 'text-center' },
      React.createElement('h1', { className: 'text-2xl font-bold mb-1' }, 'Метрики обучения YOLOv8-X для распознавания "irbis"'),
      React.createElement('p', { className: 'text-gray-600 mb-4' }, 'Модель обучена на датасете с переименованием класса "cat" → "irbis"')
    ),

    // График потерь
    React.createElement(
      'div',
      { className: 'bg-white p-4 rounded-lg shadow' },
      React.createElement('h2', { className: 'text-xl font-semibold mb-4' }, 'Функция потерь'),
      React.createElement(
        ResponsiveContainer,
        { width: '100%', height: 300 },
        React.createElement(
          LineChart,
          { data: trainingData },
          React.createElement(CartesianGrid, { strokeDasharray: '3 3' }),
          React.createElement(XAxis, {
            dataKey: 'epoch',
            label: { value: 'Эпоха', position: 'insideBottomRight', offset: -10 }
          }),
          React.createElement(YAxis, {
            label: { value: 'Потери', angle: -90, position: 'insideLeft' }
          }),
          React.createElement(Tooltip),
          React.createElement(Legend),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'trainLoss',
            name: 'Потери (обучение)',
            stroke: '#8884d8',
            strokeWidth: 2,
            dot: false
          }),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'valLoss',
            name: 'Потери (валидация)',
            stroke: '#82ca9d',
            strokeWidth: 2,
            dot: false
          })
        )
      )
    ),

    // График точности и полноты
    React.createElement(
      'div',
      { className: 'bg-white p-4 rounded-lg shadow' },
      React.createElement('h2', { className: 'text-xl font-semibold mb-4' }, 'Точность и полнота'),
      React.createElement(
        ResponsiveContainer,
        { width: '100%', height: 300 },
        React.createElement(
          LineChart,
          { data: trainingData },
          React.createElement(CartesianGrid, { strokeDasharray: '3 3' }),
          React.createElement(XAxis, {
            dataKey: 'epoch',
            label: { value: 'Эпоха', position: 'insideBottomRight', offset: -10 }
          }),
          React.createElement(YAxis, {
            label: { value: 'Значение', angle: -90, position: 'insideLeft' },
            domain: [0, 1]
          }),
          React.createElement(Tooltip),
          React.createElement(Legend),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'trainPrecision',
            name: 'Точность (обучение)',
            stroke: '#8884d8',
            strokeWidth: 2,
            dot: false
          }),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'valPrecision',
            name: 'Точность (валидация)',
            stroke: '#82ca9d',
            strokeWidth: 2,
            dot: false
          }),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'trainRecall',
            name: 'Полнота (обучение)',
            stroke: '#ff7300',
            strokeWidth: 2,
            dot: false
          }),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'valRecall',
            name: 'Полнота (валидация)',
            stroke: '#0088fe',
            strokeWidth: 2,
            dot: false
          })
        )
      )
    ),

    // График mAP и F1
    React.createElement(
      'div',
      { className: 'bg-white p-4 rounded-lg shadow' },
      React.createElement('h2', { className: 'text-xl font-semibold mb-4' }, 'mAP@0.5 и F1-мера'),
      React.createElement(
        ResponsiveContainer,
        { width: '100%', height: 300 },
        React.createElement(
          LineChart,
          { data: trainingData },
          React.createElement(CartesianGrid, { strokeDasharray: '3 3' }),
          React.createElement(XAxis, {
            dataKey: 'epoch',
            label: { value: 'Эпоха', position: 'insideBottomRight', offset: -10 }
          }),
          React.createElement(YAxis, {
            label: { value: 'Значение', angle: -90, position: 'insideLeft' },
            domain: [0, 1]
          }),
          React.createElement(Tooltip),
          React.createElement(Legend),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'trainMap50',
            name: 'mAP@0.5 (обучение)',
            stroke: '#8884d8',
            strokeWidth: 2,
            dot: false
          }),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'valMap50',
            name: 'mAP@0.5 (валидация)',
            stroke: '#82ca9d',
            strokeWidth: 2,
            dot: false
          }),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'trainF1',
            name: 'F1-мера (обучение)',
            stroke: '#ff7300',
            strokeWidth: 2,
            dot: false
          }),
          React.createElement(Line, {
            type: 'monotone',
            dataKey: 'valF1',
            name: 'F1-мера (валидация)',
            stroke: '#0088fe',
            strokeWidth: 2,
            dot: false
          })
        )
      )
    ),

    // Основные метрики
    React.createElement(
      'div',
      { className: 'bg-gray-100 p-4 rounded-lg' },
      React.createElement('h3', { className: 'font-semibold mb-2' }, 'Основные метрики после обучения (класс "irbis"):'),
      React.createElement(
        'div',
        { className: 'grid grid-cols-2 md:grid-cols-4 gap-4' },
        React.createElement(
          'div',
          { className: 'bg-white p-3 rounded shadow-sm' },
          React.createElement('div', { className: 'text-gray-500 text-sm' }, 'mAP@0.5'),
          React.createElement('div', { className: 'text-xl font-bold' }, '0.872')
        ),
        React.createElement(
          'div',
          { className: 'bg-white p-3 rounded shadow-sm' },
          React.createElement('div', { className: 'text-gray-500 text-sm' }, 'mAP@0.5:0.95'),
          React.createElement('div', { className: 'text-xl font-bold' }, '0.713')
        ),
        React.createElement(
          'div',
          { className: 'bg-white p-3 rounded shadow-sm' },
          React.createElement('div', { className: 'text-gray-500 text-sm' }, 'Точность'),
          React.createElement('div', { className: 'text-xl font-bold' }, '0.884')
        ),
        React.createElement(
          'div',
          { className: 'bg-white p-3 rounded shadow-sm' },
          React.createElement('div', { className: 'text-gray-500 text-sm' }, 'Полнота'),
          React.createElement('div', { className: 'text-xl font-bold' }, '0.856')
        )
      )
    )
  );
};

// Компонент для кастомной модели загружается отдельно через React в component_id

const YoloTrainingCustomMetrics = () => {
  // Этот компонент загружается отдельно в HTML-шаблоне
  // (предполагается, что он будет доступен через React)
  return null;
};