<?php

use Livewire\Volt\Component;
use Livewire\WithFileUploads;
use Illuminate\Support\Facades\Http;

new class extends Component
{
    use WithFileUploads;

    public $images = [];
    public string $instancePrompt = '';
    public int $maxTrainSteps = 800;
    public float $learningRate = 0.0001;
    
    public string $status = 'idle'; 
    public array $logs = [];
    public string $message = '';
    public bool $isCheckingStatus = false;

    public function startTraining()
    {
        $this->validate([
            'images' => 'required|array|min:1',
            'instancePrompt' => 'required|string|min:2',
            'maxTrainSteps' => 'required|integer|min:100|max:2000',
            'learningRate' => 'required|numeric',
        ]);

        $this->status = 'uploading';
        $this->message = '画像をAIエンジンに転送中...';

        try {
            // 1. 画像をアップロード
            $request = Http::asMultipart();
            foreach ($this->images as $image) {
                $request->attach('files', file_get_contents($image->getRealPath()), $image->getClientOriginalName());
            }
            
            $aiUrl = env('AI_ENGINE_URL', 'http://ai:8000');
            $uploadResponse = $request->post($aiUrl . '/train/upload');

            if (!$uploadResponse->successful()) {
                throw new \Exception('画像のアップロードに失敗しました: ' . $uploadResponse->body());
            }

            // 2. 学習開始
            $startResponse = Http::post($aiUrl . '/train/start', [
                'instance_prompt' => $this->instancePrompt,
                'max_train_steps' => $this->maxTrainSteps,
                'learning_rate' => $this->learningRate,
            ]);

            if (!$startResponse->successful()) {
                throw new \Exception('学習の開始に失敗しました: ' . $startResponse->body());
            }

            $this->status = 'training';
            $this->message = '学習を開始しました。完了までブラウザを閉じずにお待ちください（または後で確認してください）。';
            $this->isCheckingStatus = true;

        } catch (\Exception $e) {
            $this->status = 'error';
            $this->message = 'エラーが発生しました: ' . $e->getMessage();
        }
    }

    public function checkStatus()
    {
        if (!$this->isCheckingStatus) return;

        try {
            $aiUrl = env('AI_ENGINE_URL', 'http://ai:8000');
            $response = Http::get($aiUrl . '/train/status');

            if ($response->successful()) {
                $data = $response->json();
                $this->logs = $data['last_log'] ?? [];
                
                if ($data['status'] === 'finished/error' && !empty($this->logs)) {
                    // ログの最後の方を見て成功か失敗か判断（簡易的）
                    $lastLog = implode('', $this->logs);
                    if (str_contains($lastLog, 'Steps: 100%')) {
                        $this->status = 'finished';
                        $this->message = '学習が正常に完了しました！🎉';
                        $this->isCheckingStatus = false;
                    } else if (str_contains($lastLog, 'Error') || str_contains($lastLog, 'error')) {
                        $this->status = 'error';
                        $this->message = '学習中にエラーが発生した可能性があります。ログを確認してください。';
                        $this->isCheckingStatus = false;
                    }
                }
            }
        } catch (\Exception $e) {
            // サイレントに無視して再試行
        }
    }
}; ?>

<div wire:poll.3s="checkStatus">
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <!-- Input Panel -->
        <div class="lg:col-span-1 space-y-6">
            <div class="glass rounded-3xl p-6 border border-white/10">
                <h2 class="text-xl font-bold font-display mb-4 flex items-center">
                    <span class="mr-2">📁</span> 学習データの準備
                </h2>
                
                <div class="space-y-4">
                    <!-- Image Upload -->
                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-2">学習画像 (10-20枚推奨)</label>
                        <div 
                            class="relative border-2 border-dashed border-white/10 rounded-2xl p-8 transition-all hover:border-blue-500/50 group cursor-pointer"
                            x-on:click="$refs.fileInput.click()"
                        >
                            <input type="file" wire:model="images" multiple class="hidden" x-ref.fileInput>
                            <div class="text-center">
                                <span class="text-3xl mb-2 block">📸</span>
                                <p class="text-sm text-gray-400 group-hover:text-gray-300">クリックして画像を選択</p>
                                <p class="text-xs text-gray-500 mt-1">PNG, JPG (512x512以上推奨)</p>
                            </div>
                        </div>
                        @error('images') <span class="text-red-400 text-xs mt-1">{{ $message }}</span> @enderror
                        
                        @if ($images)
                            <div class="mt-4 grid grid-cols-4 gap-2">
                                @foreach ($images as $image)
                                    <div class="aspect-square rounded-lg overflow-hidden border border-white/10 bg-white/5">
                                        <img src="{{ $image->temporaryUrl() }}" class="w-full h-full object-cover">
                                    </div>
                                @endforeach
                            </div>
                        @endif
                    </div>

                    <!-- Instance Prompt -->
                    <div>
                        <label class="block text-sm font-medium text-gray-400 mb-2">トリガーワード (合言葉)</label>
                        <input 
                            type="text" 
                            wire:model="instancePrompt" 
                            placeholder="例: my_character_style"
                            class="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                        >
                        <p class="text-[10px] text-gray-500 mt-1">AIがこの言葉に反応して学習内容を反映します。</p>
                        @error('instancePrompt') <span class="text-red-400 text-xs mt-1">{{ $message }}</span> @enderror
                    </div>

                    <!-- Advanced Settings Toggle -->
                    <div x-data="{ open: false }">
                        <button @click="open = !open" type="button" class="text-xs text-gray-500 hover:text-gray-300 flex items-center transition-colors">
                            <span x-text="open ? '▼' : '▶'" class="mr-1"></span> 詳細設定
                        </button>
                        <div x-show="open" x-cloak class="mt-4 space-y-4 pt-4 border-t border-white/5">
                            <div>
                                <label class="block text-xs text-gray-500 mb-1">学習ステップ数 (Max Steps)</label>
                                <input type="number" wire:model="maxTrainSteps" class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm">
                            </div>
                            <div>
                                <label class="block text-xs text-gray-500 mb-1">学習率 (Learning Rate)</label>
                                <input type="text" wire:model="learningRate" class="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm">
                            </div>
                        </div>
                    </div>

                    <button 
                        wire:click="startTraining"
                        wire:loading.attr="disabled"
                        class="w-full py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white rounded-2xl font-bold shadow-lg shadow-blue-900/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed mt-4"
                    >
                        <span wire:loading.remove wire:target="startTraining">学習を開始する ✨</span>
                        <span wire:loading wire:target="startTraining">準備中... ⏳</span>
                    </button>
                </div>
            </div>
        </div>

        <!-- Status & Logs Panel -->
        <div class="lg:col-span-2 space-y-6">
            <div class="glass rounded-3xl p-6 border border-white/10 h-full flex flex-col">
                <h2 class="text-xl font-bold font-display mb-4 flex items-center">
                    <span class="mr-2">📊</span> 学習ステータス
                </h2>

                <div class="flex-1 flex flex-col">
                    <!-- Status Badge -->
                    <div class="mb-6 p-4 rounded-2xl @if($status === 'training') bg-blue-500/10 border-blue-500/20 @elseif($status === 'finished') bg-green-500/10 border-green-500/20 @elseif($status === 'error') bg-red-500/10 border-red-500/20 @else bg-white/5 border-white/10 @endif border">
                        <div class="flex items-center justify-between">
                            <span class="font-medium">状態: {{ [
                                'idle' => '待機中',
                                'uploading' => 'アップロード中...',
                                'training' => '学習実行中...',
                                'finished' => '完了！',
                                'error' => 'エラー発生'
                            ][$status] }}</span>
                            @if($status === 'training')
                                <div class="flex space-x-1">
                                    <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                                    <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                                    <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></div>
                                </div>
                            @endif
                        </div>
                        <p class="text-sm text-gray-400 mt-1">{{ $message }}</p>
                    </div>

                    <!-- Logs View -->
                    <div class="flex-1 bg-black/40 rounded-2xl p-4 font-mono text-xs overflow-y-auto space-y-1 min-h-[300px] border border-white/5">
                        @forelse($logs as $log)
                            <div class="text-gray-400">{{ $log }}</div>
                        @empty
                            <div class="text-gray-600 italic">学習を開始するとここにログが表示されます...</div>
                        @endforelse
                    </div>

                    @if($status === 'finished')
                        <div class="mt-6 p-4 glass rounded-2xl border-green-500/30 border">
                            <p class="text-sm text-green-400 font-medium">✨ 学習が完了しました！</p>
                            <p class="text-xs text-gray-400 mt-1">生成画面に戻って、プロンプトに「{{ $instancePrompt }}」を含めて生成してみてください。</p>
                            <p class="text-[10px] text-gray-500 mt-2">※ LoRAモデルの自動適用は現在実装中です。</p>
                        </div>
                    @endif
                </div>
            </div>
        </div>
    </div>
</div>
