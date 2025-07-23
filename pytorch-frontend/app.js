const { createApp } = Vue;

createApp({
  data() {
    return {
      currentView: 'lessons',
      lessons: [],
      currentLesson: null,
      currentStepIndex: 0,
      userCode: '',
      output: null,
      error: null,
      hasError: false,
      isLoading: false,
      feedback: null,
      showHint: false,
      currentHint: '',
      baseUrl: 'http://localhost:5000/api'
    };
  },

  computed: {
    currentStep() {
      return this.currentLesson?.steps?.[this.currentStepIndex] ?? null;
    },
    progressPercentage() {
      if (!this.currentLesson?.steps?.length) return 0;
      return Math.round(((this.currentStepIndex + 1) / this.currentLesson.steps.length) * 100);
    }
  },

  mounted() {
    this.loadLessons();
  },

  methods: {
    async loadLessons() {
      const res = await axios.get(`${this.baseUrl}/lessons`);
      this.lessons = res.data;
    },
    async selectLesson(id) {
      const res = await axios.get(`${this.baseUrl}/lessons/${id}`);
      this.currentLesson = res.data;
      this.currentView = 'lesson';
      this.currentStepIndex = 0;
      this.userCode = this.currentStep?.starter_code || '';
      this.feedback = null;
    },
    backToLessons() {
      this.currentView = 'lessons';
      this.currentLesson = null;
      this.userCode = '';
    },
    nextStep() {
      if (this.currentStepIndex < this.currentLesson.steps.length - 1) {
        this.currentStepIndex++;
        this.userCode = this.currentStep.starter_code || '';
        this.feedback = null;
      }
    },
    previousStep() {
      if (this.currentStepIndex > 0) {
        this.currentStepIndex--;
        this.userCode = this.currentStep.starter_code || '';
        this.feedback = null;
      }
    },
    async runCode() {
      this.isLoading = true;
      this.output = null;
      this.error = null;
      try {
        const res = await axios.post(`${this.baseUrl}/run_code`, {
          code: this.userCode
        });
        this.output = res.data.output;
        this.hasError = !res.data.success;
        this.error = res.data.error;
      } catch (err) {
        this.output = '';
        this.error = err.message;
        this.hasError = true;
      } finally {
        this.isLoading = false;
      }
    },
    async checkSolution() {
      const res = await axios.post(`${this.baseUrl}/check_solution`, {
        code: this.userCode,
        solution: this.currentStep.solution_code
      });
      this.feedback = {
        correct: res.data.correct,
        message: res.data.message
      };
    },
    toggleHint() {
      this.showHint = !this.showHint;
      if (this.showHint) {
        this.currentHint = this.currentStep.hint || 'No hint available';
      }
    },
    showSolution() {
      this.userCode = this.currentStep.solution_code || '';
    },
    resetCode() {
      this.userCode = this.currentStep.starter_code || '';
      this.feedback = null;
      this.output = null;
      this.error = null;
    }
  }
}).mount('#app');
