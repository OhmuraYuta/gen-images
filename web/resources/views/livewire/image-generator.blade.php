<?php

use Livewire\Volt\Component;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;

new class extends Component
{
    public string $prompt = '';
    public string $negativePrompt = 'ugly, deformed, disfigured, poor quality, blurry, low res, bad anatomy, bad hands, text, error, missing fingers, bad legs,';
    public float $guidanceScale = 7.5;
    public int $numInferenceSteps = 30;
    public int $width = 1024;
    public int $height = 1024;
    public int $seed = -1;
    public bool $isRandomSeed = true;
    
    public string $imagePath = '';
    public string $translatedPrompt = '';
    public bool $isGenerating = false;
    public bool $showSettings = false;
    public string $error = '';

    public function generate()
    {
        $this->validate([
            'prompt' => 'required|string|min:3',
            'negativePrompt' => 'nullable|string',
            'guidanceScale' => 'required|numeric|min:1|max:20',
            'numInferenceSteps' => 'required|integer|min:1|max:100',
            'width' => 'required|integer|multiple_of:8',
            'height' => 'required|integer|multiple_of:8',
        ]);

        $this->isGenerating = true;
        $this->error = '';
        $this->imagePath = '';
        $this->translatedPrompt = '';

        try {
            $response = Http::timeout(400)->post(env('AI_ENGINE_URL', 'http://ai:8000') . '/generate', [
                'prompt' => $this->prompt,
                'negative_prompt' => $this->negativePrompt,
                'guidance_scale' => $this->guidanceScale,
                'num_inference_steps' => $this->numInferenceSteps,
                'width' => $this->width,
                'height' => $this->height,
                'seed' => $this->isRandomSeed ? -1 : $this->seed,
            ]);

            if ($response->successful()) {
                if ($response->hasHeader('X-Used-Seed')) {
                    $this->seed = (int) $response->header('X-Used-Seed');
                }

                $filename = 'generated/latest.png';
                Storage::disk('public')->put($filename, $response->body());
                $this->imagePath = Storage::url($filename) . '?t=' . time();
                $this->translatedPrompt = $response->header('X-Translated-Prompt') ?? '';
            } else {
                $this->error = 'AIエンジン側でエラーが発生しました。: ' . $response->body();
            }
        } catch (\Exception $e) {
            $this->error = 'AIエンジンへの接続に失敗しました: ' . $e->getMessage();
        }

        $this->isGenerating = false;
    }

    public function toggleSettings()
    {
        $this->showSettings = !$this->showSettings;
    }

    public function setAspectRatio(string $ratio)
    {
        switch ($ratio) {
            case '1:1': $this->width = 1024; $this->height = 1024; break;
            case '3:4': $this->width = 768; $this->height = 1024; break;
            case '16:9': $this->width = 1024; $this->height = 576; break;
        }
    }
};
?>

<div class="grid lg:grid-cols-2 gap-16 items-start">
    <!-- Left Column: Controls -->
    <div class="space-y-8">
        <div>
            <h1 class="text-5xl lg:text-7xl font-bold leading-tight mb-6">
                Visualize your <br>
                <span class="text-transparent bg-clip-text bg-gradient-to-r from-[#f53003] to-orange-400">imagination</span>
            </h1>
            <p class="text-lg text-[#706f6c] dark:text-[#A1A09A] max-w-lg leading-relaxed">
                あなたの想像力を、ローカルGPUのポテンシャルで鮮やかなリアリティへ。
                最新の Stable Diffusion XL (SDXL) を基盤とした画像生成システムが、比類なき高精細なビジュアル体験を数秒で提供します。
            </p>
        </div>

        <div class="p-8 glass rounded-2xl shadow-2xl space-y-6">
            <div class="flex items-center justify-between mb-2">
                <label for="prompt" class="block text-sm font-medium text-[#1b1b18] dark:text-[#EDEDEC]">プロンプト</label>
                <button wire:click="toggleSettings" class="text-xs font-medium text-[#f53003] hover:underline flex items-center gap-1">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 12h7.5" />
                    </svg>
                    こだわり設定
                </button>
            </div>
            
            <div class="flex gap-2">
                <input 
                    type="text" 
                    id="prompt" 
                    wire:model="prompt" 
                    class="flex-1 px-4 py-3 border rounded-xl border-[#19140035] dark:border-[#3E3E3A] bg-white dark:bg-[#161615] text-[#1b1b18] dark:text-[#EDEDEC] focus:outline-none focus:ring-2 focus:ring-[#f53003]"
                    placeholder="例: サイバーパンクなネオン輝く未来都市、傑作、高精細"
                    @keydown.enter="if (!$event.isComposing) $wire.generate()"
                >
                <button 
                    wire:click="generate" 
                    wire:loading.attr="disabled"
                    class="px-8 py-3 bg-[#1b1b18] dark:bg-[#eeeeec] text-white dark:text-[#1C1C1A] rounded-xl font-bold hover:opacity-90 transition-all disabled:opacity-50 shadow-lg active:scale-95"
                >
                    <div wire:loading.remove wire:target="generate">生成</div>
                    <div wire:loading wire:target="generate" class="flex items-center gap-2">
                        <div class="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                    </div>
                </button>
            </div>
            
            @if($translatedPrompt)
                <div class="mt-2 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl text-xs text-blue-700 dark:text-blue-300 flex items-start gap-2 leading-relaxed">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5 mt-0.5 shrink-0">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M10.5 21l5.25-11.25L21 21m-9-3h7.5M3 5.621a48.474 48.474 0 016-.371m0 0c1.12 0 2.233.038 3.334.114M9 5.25V3m3.334 2.364C11.176 10.658 7.69 15.08 3 17.502m9.334-12.138A44.26 44.26 0 0115.5 12m-2 2.138c1.139 1.694 2.828 2.93 4.786 3.558L10 11.214" />
                    </svg>
                    <span>AIによる解釈 (英語): {{ $translatedPrompt }}</span>
                </div>
            @endif

            @error('prompt') <span class="text-xs text-[#f53003] mt-1">{{ $message }}</span> @enderror

            <!-- Settings Panel (Inside Left Column) -->
            <div x-show="$wire.showSettings" x-transition class="p-5 border border-[#19140035] dark:border-[#3E3E3A] rounded-2xl bg-neutral-50 dark:bg-neutral-900/50 space-y-4 shadow-inner">
                <div>
                    <label class="block mb-1 text-xs font-bold uppercase tracking-wider opacity-60">ネガティブプロンプト</label>
                    <textarea 
                        wire:model="negativePrompt" 
                        rows="2"
                        class="w-full px-3 py-2 text-sm border rounded-xl border-[#19140035] dark:border-[#3E3E3A] bg-white dark:bg-[#161615] focus:outline-none"
                    ></textarea>
                </div>
                <div class="grid grid-cols-2 gap-4 text-xs">
                    <div>
                        <label class="block mb-1 font-bold uppercase tracking-wider opacity-60">Guidance ({{ $guidanceScale }})</label>
                        <input type="range" wire:model.live="guidanceScale" min="1" max="20" step="0.5" class="w-full accent-[#f53003]">
                    </div>
                    <div>
                        <label class="block mb-1 font-bold uppercase tracking-wider opacity-60">Steps ({{ $numInferenceSteps }})</label>
                        <input type="range" wire:model.live="numInferenceSteps" min="1" max="100" class="w-full accent-[#f53003]">
                    </div>
                </div>
                <div class="pt-2 border-t border-[#19140010] dark:border-[#3E3E3A50]" x-data="{ isRandom: @entangle('isRandomSeed') }">
                    <div class="flex items-center justify-between mb-2">
                        <label class="text-xs font-bold uppercase tracking-wider opacity-60">シード値 (Seed)</label>
                        <div class="flex items-center gap-2 cursor-pointer group" @click="isRandom = !isRandom">
                            <span class="text-[10px]" :class="isRandom ? 'opacity-50' : 'opacity-100 font-bold'" x-text="isRandom ? 'ランダム' : '固定中'"></span>
                            <div class="w-8 h-4 rounded-full relative transition-colors" :class="isRandom ? 'bg-neutral-300 dark:bg-neutral-700' : 'bg-[#f53003]'">
                                <div class="absolute top-0.5 left-0.5 w-3 h-3 bg-white rounded-full transition-transform" :class="isRandom ? 'translate-x-0' : 'translate-x-4'"></div>
                            </div>
                        </div>
                    </div>
                    <div class="flex gap-2">
                        <input type="number" wire:model="seed" x-bind:disabled="isRandom" class="flex-1 px-3 py-1.5 text-sm border rounded-lg border-[#19140035] bg-white dark:bg-black focus:outline-none disabled:opacity-30">
                        <button @click="isRandom = true" class="px-3 border rounded-lg border-[#19140035] hover:bg-black/5 transition-all active:scale-95">🎲</button>
                    </div>
                </div>
            </div>

            @if($error)
                <div class="p-4 text-sm text-[#f53003] bg-[#fff2f2] dark:bg-[#1D0002] border border-[#f53003] rounded-xl flex items-start gap-2">
                    <span>{{ $error }}</span>
                </div>
            @endif
        </div>
    </div>

    <!-- Right Column: Dynamic Showcase -->
    <div class="hidden lg:block relative sticky top-12">
        <div class="relative overflow-hidden rounded-3xl shadow-2xl border border-white/20 aspect-[4/5] bg-neutral-200 dark:bg-neutral-900 flex items-center justify-center">
            @if($isGenerating)
                <!-- Generating State -->
                <div class="flex flex-col items-center gap-6 z-10">
                    <div class="w-16 h-16 border-4 border-[#1b1b18] dark:border-[#eeeeec] border-t-transparent rounded-full animate-spin"></div>
                    <div class="text-center">
                        <p class="text-xl font-bold animate-pulse">精細な画像を描き込み中...</p>
                        <p class="text-xs opacity-60 mt-2">RTX 4070 Powered Acceleration</p>
                    </div>
                </div>
                <div class="absolute inset-0 bg-neutral-100/10 backdrop-blur-sm animate-pulse"></div>
            @elseif($imagePath)
                <!-- Success State -->
                <img src="{{ $imagePath }}" alt="Generated Artifact" class="w-full h-full object-contain p-4 transition-all duration-500">
                <div class="absolute bottom-4 left-4 right-4 glass px-5 py-3 rounded-2xl border border-white/10 flex justify-between items-center group shadow-2xl">
                    <div class="overflow-hidden">
                        <p class="text-[10px] font-bold mb-0.5 uppercase tracking-widest opacity-40">Composition Result</p>
                        <p class="text-sm font-medium truncate">✨ {{ Str::limit($prompt, 30) }}</p>
                    </div>
                    <div class="flex items-center gap-3">
                        <div class="h-8 w-[1px] bg-white/10"></div>
                        <a href="{{ $imagePath }}" download class="p-2.5 bg-white/10 hover:bg-[#f53003] rounded-xl transition-all active:scale-90 group/btn">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2.5" stroke="currentColor" class="w-5 h-5 transition-transform group-hover/btn:-translate-y-0.5">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                            </svg>
                        </a>
                    </div>
                </div>
            @else
                <!-- Idle State -->
                <div class="absolute inset-0 flex flex-col items-center justify-center opacity-20 select-none">
                    <span class="text-8xl font-black italic tracking-tighter uppercase opacity-10">AI POWERED</span>
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1" stroke="currentColor" class="w-24 h-24 mt-8">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V6.75zm.375 0a.375 3.375 0 11-.75 0 .375 3.375 0 01.75 0z" />
                    </svg>
                </div>
                <div class="absolute bottom-8 left-8 right-8 glass p-6 rounded-2xl border border-white/10">
                    <p class="text-sm font-semibold mb-1 uppercase tracking-wider opacity-50">Recent Artifact</p>
                    <p class="text-xl font-medium italic">🎨 Awaiting your imagination...</p>
                </div>
            @endif
        </div>
        
        <!-- Mini Badge -->
        <div class="absolute -top-6 -right-6 px-6 py-3 bg-[#f53003] text-white rounded-full font-bold shadow-xl rotate-12 flex items-center gap-2 group cursor-help transition-all hover:rotate-6 active:scale-95">
            <span>NVIDIA CUDA</span>
            <div class="w-2 h-2 bg-green-400 rounded-full animate-ping"></div>
        </div>
    </div>
</div>