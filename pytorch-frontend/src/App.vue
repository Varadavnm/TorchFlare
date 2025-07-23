<template>
  <div class="app">
    <h1>PyTorch Learning Platform</h1>

    <div v-if="lessons.length">
      <button 
        v-for="(lesson, index) in lessons" 
        :key="index" 
        @click="selectLesson(index)">
        {{ lesson.title }}
      </button>
    </div>

    <div v-if="selectedLesson">
      <h2>{{ selectedLesson.title }}</h2>
      <p>{{ selectedLesson.description }}</p>
      <textarea v-model="codeInput"></textarea>
      <button @click="runCode">Run</button>
      <pre>{{ output }}</pre>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  data() {
    return {
      lessons: [],
      selectedLesson: null,
      codeInput: '',
      output: '',
    };
  },
  mounted() {
    axios.get('http://localhost:5000/api/lessons')
      .then(response => {
        this.lessons = response.data.lessons;
      });
  },
  methods: {
    selectLesson(index) {
      this.selectedLesson = this.lessons[index];
      this.codeInput = this.selectedLesson.starter_code || '';
    },
    runCode() {
      axios.post('http://localhost:5000/api/run_code', {
        code: this.codeInput,
      }).then(response => {
        this.output = response.data.output;
      }).catch(error => {
        this.output = error.response?.data?.error || 'Error running code';
      });
    },
  }
};
</script>

<style>
textarea {
  width: 100%;
  height: 200px;
  margin-top: 10px;
}
</style>
